"""Command-line interface: ``cryo2dclass {train, evaluate, visualize, pipeline}``.

Run ``cryo2dclass <command> --help`` for all options. Multi-GPU training::

    accelerate launch -m cryo2dclass train --stage 1 --data particles.mrcs --out runs/stage1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

from . import __version__
from .config import TrainConfig, add_config_arguments, config_to_yaml, load_yaml, namespace_overrides, resolve_config

log = logging.getLogger("cryo2dclass")

EPILOG_TRAIN = """
Precedence: stage preset < --config YAML < command-line flags.
Pixel values of the presets are tuned for 128-px boxes and rescaled to your box size.

examples:
  cryo2dclass train --stage 1 --data particles.mrcs --out runs/stage1
  cryo2dclass train --stage 2 --data particles.mrcs --out runs/stage2 --init-from runs/stage1
  cryo2dclass train --stage 3 --data particles.mrcs --out runs/stage3 --init-from runs/stage2
  cryo2dclass train --resume --out runs/stage2 --n-rounds 15      # continue / extend a run
  cryo2dclass train --stage 2 --data particles.mrcs --print-config # show the resolved settings
"""


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0


def _setup_logging(verbose: bool = False) -> None:
    level = (logging.DEBUG if verbose else logging.INFO) if _is_main_process() else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout)
    log.setLevel(level)


# --------------------------------------------------------------------------- config resolution

def build_train_config(args: argparse.Namespace) -> TrainConfig:
    """Resolve the configuration of a ``train`` invocation."""
    from .checkpoint import load_run_config

    overrides = load_yaml(args.config) if args.config else {}
    overrides.update(namespace_overrides(args))

    if args.resume:
        if not overrides.get("out"):
            raise SystemExit("--resume needs --out <run directory>")
        base = load_run_config(overrides["out"]).to_dict()
        base.update(overrides)
        return resolve_config(base["stage"], {k: v for k, v in base.items() if k != "stage"},
                              box_size=_box(base.get("data")))

    stage = overrides.pop("stage", None)
    if stage is None:
        raise SystemExit("Please choose a stage: --stage 1, 2 or 3 (or set 'stage' in --config)")
    data = overrides.get("data")
    if data is None and not args.print_config:
        raise SystemExit("--data is required")
    if overrides.get("out") is None and not args.print_config:
        raise SystemExit("--out is required")
    return resolve_config(stage, overrides, box_size=_box(data) if data else 128)


def _box(data: Optional[str]) -> int:
    from .data import stack_shape

    return stack_shape(data)[-1] if data else 128


def _labels(args: argparse.Namespace):
    from .data import labels_from_counts, load_labels

    if getattr(args, "labels", None) and getattr(args, "label_counts", None):
        raise SystemExit("Use either --labels or --label-counts, not both")
    if getattr(args, "labels", None):
        return load_labels(args.labels)
    if getattr(args, "label_counts", None):
        return labels_from_counts(args.label_counts)
    return None


# --------------------------------------------------------------------------- commands

def cmd_train(args: argparse.Namespace) -> None:
    cfg = build_train_config(args)
    if args.print_config:
        print(config_to_yaml(cfg), end="")
        return
    from .train import train

    if not args.resume and os.path.exists(os.path.join(cfg.out, "checkpoints", "last.pt")):
        raise SystemExit(f"{cfg.out} already contains a trained run. Use --resume to continue it, "
                         "or choose another --out.")
    train(cfg, resume=args.resume)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .evaluate import evaluate

    evaluate(args.run, checkpoint=args.checkpoint, data=args.data, n_clusters=args.n_clusters,
             labels=_labels(args), pca_dims=args.pca_dims, out_dir=args.out,
             batch_size=args.batch_size, device=args.device, seed=args.seed)


def cmd_visualize(args: argparse.Namespace) -> None:
    from .plotting import visualize

    visualize(args.run, checkpoint=args.checkpoint, figures=args.figures, eval_dir=args.eval_dir,
              particle_ids=args.particles, map_size=args.map_size, out_dir=args.out,
              data=args.data, device=args.device)


def cmd_pipeline(args: argparse.Namespace) -> None:
    """All stages, then evaluate + visualize the last one."""
    from accelerate import Accelerator

    from .checkpoint import load_run_config
    from .data import stack_shape
    from .train import train

    n_particles, box, _ = stack_shape(args.data)
    labels = _labels(args)  # fail early on bad label files, not after hours of training
    if labels is not None and len(labels) != n_particles:
        raise SystemExit(f"{len(labels)} labels given for {n_particles} particles")
    common = namespace_overrides(args)
    for key in ("stage", "out", "init_from", "init_checkpoint"):
        common.pop(key, None)
    accelerator = Accelerator()
    previous = None
    for stage in args.stages:
        run_dir = os.path.join(args.out, f"stage{stage}")
        done = os.path.exists(os.path.join(run_dir, "checkpoints", "last.pt"))
        if done and not args.resume:
            raise SystemExit(f"{run_dir} already exists. Use --resume to continue the pipeline.")
        if done:
            cfg = load_run_config(run_dir)
            cfg = resolve_config(stage, {**{k: v for k, v in cfg.to_dict().items() if k != "stage"},
                                         **common}, box)
        else:
            cfg = resolve_config(stage, {**common, "out": run_dir, "init_from": previous}, box)
        log.info("========== stage %d -> %s", stage, run_dir)
        train(cfg, resume=done, accelerator=accelerator)
        previous = run_dir

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        from .evaluate import evaluate
        from .plotting import visualize

        evaluate(previous, n_clusters=args.n_clusters, labels=labels, pca_dims=args.pca_dims)
        visualize(previous)
        log.info("Pipeline finished. Final results: %s", previous)


# --------------------------------------------------------------------------- parser

def _add_eval_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("clustering")
    g.add_argument("--n-clusters", type=int, default=2, help="Number of K-means clusters (default: 2).")
    g.add_argument("--pca-dims", type=int, default=2,
                   help="Cluster in this many principal components of the latent space; 0 = use the "
                        "full latent space (default: 2).")
    g.add_argument("--labels", help="Ground-truth labels (.npy/.txt/.csv, one integer per particle) "
                                    "to report accuracy, ARI and NMI.")
    g.add_argument("--label-counts", type=int, nargs="+", metavar="N",
                   help="Alternative to --labels for stacks sorted by class, e.g. --label-counts 5000 5000.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryo2dclass",
        description="Unsupervised 2D classification of cryo-EM/ET particles with a rotation-invariant "
                    "autoencoder and built-in pose search.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # train
    p = sub.add_parser("train", help="Train one stage.", epilog=EPILOG_TRAIN,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML file with configuration values (keys as in config.yaml).")
    p.add_argument("--resume", action="store_true",
                   help="Continue the run in --out from its last checkpoint.")
    p.add_argument("--print-config", action="store_true", help="Print the resolved configuration and exit.")
    add_config_arguments(p)
    p.set_defaults(func=cmd_train)

    # evaluate
    p = sub.add_parser("evaluate", help="Embed, cluster and summarise a trained run.")
    p.add_argument("run", help="Run directory created by `train`.")
    p.add_argument("--checkpoint", default="best", help="best, last or round_NNN (default: best).")
    p.add_argument("--data", help="Particle stack (default: the one used for training).")
    p.add_argument("--out", help="Output directory (default: <run>/eval_<checkpoint>).")
    p.add_argument("--batch-size", type=int, help="Batch size (default: training batch size).")
    p.add_argument("--device", help="e.g. cuda, cuda:1 or cpu (default: cuda if available).")
    p.add_argument("--seed", type=int, default=0, help="K-means seed (default: 0).")
    _add_eval_arguments(p)
    p.set_defaults(func=cmd_evaluate)

    # visualize
    from .plotting import FIGURES

    p = sub.add_parser("visualize", help="Write PNG figures for a trained (and evaluated) run.")
    p.add_argument("run", help="Run directory created by `train`.")
    p.add_argument("--checkpoint", default="best", help="best, last or round_NNN (default: best).")
    p.add_argument("--figures", nargs="+", default=list(FIGURES), choices=FIGURES, metavar="NAME",
                   help=f"Subset of: {' '.join(FIGURES)} (default: all).")
    p.add_argument("--eval-dir", help="Evaluation directory (default: <run>/eval_<checkpoint>).")
    p.add_argument("--particles", type=int, nargs="+", metavar="ID",
                   help="Particle indices for reconstructions.png (default: 6 evenly spaced).")
    p.add_argument("--map-size", type=int, default=4, help="Latent map grid size (default: 4 -> 4x4).")
    p.add_argument("--data", help="Particle stack (default: the one used for training).")
    p.add_argument("--out", help="Output directory (default: <run>/figures).")
    p.add_argument("--device", help="e.g. cuda or cpu (default: cuda if available).")
    p.set_defaults(func=cmd_visualize)

    # pipeline
    p = sub.add_parser("pipeline", help="Run stages 1-3, then evaluate and visualize.",
                       epilog="Hyper-parameter flags apply to every stage (on top of each stage's preset).\n\n"
                              "example:\n  cryo2dclass pipeline --data particles.mrcs --out runs/exp1 "
                              "--n-clusters 2",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3], choices=(1, 2, 3),
                   help="Stages to run in order (default: 1 2 3).")
    p.add_argument("--resume", action="store_true", help="Continue an interrupted pipeline.")
    _add_eval_arguments(p)
    add_config_arguments(p, skip=("stage", "init_from", "init_checkpoint"))
    p.set_defaults(func=cmd_pipeline)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    if args.command == "pipeline" and (not getattr(args, "data", None) or not getattr(args, "out", None)):
        raise SystemExit("pipeline needs --data and --out")
    args.func(args)


if __name__ == "__main__":
    main()
