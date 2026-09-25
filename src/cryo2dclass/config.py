"""Training configuration: one flat dataclass, three stage presets.

How a configuration is resolved (later wins):

1. ``TrainConfig`` defaults (stage-independent settings),
2. the stage preset (``STAGE_PRESETS``), with pixel-valued entries rescaled
   from the 128-px reference box to the actual box size,
3. values from a YAML file passed with ``--config``,
4. flags given on the command line.

The fully resolved configuration is written to ``<run>/config.yaml`` so every
run directory is self-describing, and ``evaluate`` / ``visualize`` read it back
instead of asking the user to repeat hyper-parameters.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

import yaml

#: Box size (pixels) the stage presets were tuned for.
REFERENCE_BOX = 128

#: Preset entries measured in input-image pixels; rescaled by box / REFERENCE_BOX.
PIXEL_FIELDS = (
    "shift_step",
    "mask_radius",
    "mask_edge_width",
    "warmup_mask_radius",
    "warmup_max_shift",
)


def _f(default, help, group, **kw):
    """Shorthand for a dataclass field that carries CLI help text."""
    return field(default=default, metadata={"help": help, "group": group, **kw})


@dataclass
class TrainConfig:
    # ---------------------------------------------------------------- I/O
    data: Optional[str] = _f(None, "Particle stack: .npy array (N, H, W) or MRC stack "
                             "(.mrcs/.mrc/.st/.ali). Box size must be a multiple of 16.", "input/output",
                             metavar="PATH")
    out: Optional[str] = _f(None, "Run directory to create (checkpoints, logs, results).", "input/output",
                            metavar="DIR")
    init_from: Optional[str] = _f(None, "Run directory of the previous stage; its weights and "
                                  "particle poses initialise this run.", "input/output", metavar="DIR")
    init_checkpoint: str = _f("best", "Checkpoint of --init-from to start from: best, last or round_NNN.",
                              "input/output", metavar="NAME")
    normalize: bool = _f(False, "Normalise every particle to zero mean / unit std before training.",
                         "input/output")

    # ---------------------------------------------------------------- stage / model
    stage: int = _f(1, "Training stage (1, 2 or 3); selects the preset below.", "model", choices=(1, 2, 3))
    half_res: bool = _f(True, "Work on 2x-binned images (stage 1: fast, coarse, global pose search).", "model")
    boost_decoder: bool = _f(False, "Use the learned 2x up-sampling decoder instead of pixel "
                             "duplication (stage 3).", "model")
    latent_dim: int = _f(8, "Dimension of the latent (embedding) space.", "model")
    latent_noise: float = _f(0.1, "Std of Gaussian noise added to latent vectors during training.", "model")
    latent_dropout: float = _f(0.3, "Dropout rate before the last latent layer.", "model")
    latent_norm_max: float = _f(20.0, "Latent vectors with a larger L2 norm are penalised.", "model")
    latent_reg_weight: float = _f(0.01, "Weight of the latent-norm penalty.", "model")
    lowpass_size: int = _f(3, "Size of the Gaussian low-pass kernel applied to inputs and "
                           "reconstructions (odd integer).", "model")

    # ---------------------------------------------------------------- optimisation
    n_rounds: int = _f(10, "Refinement rounds. One round = several training epochs followed by a "
                       "pose search over all particles.", "optimisation")
    epochs_per_round: int = _f(10, "Training epochs (passes over the particles) per round.", "optimisation")
    batch_size: int = _f(128, "Batch size per GPU.", "optimisation")
    lr: float = _f(5e-4, "AdamW learning rate.", "optimisation")
    weight_decay: float = _f(0.01, "AdamW weight decay.", "optimisation")
    seed: int = _f(42, "Random seed.", "optimisation")

    # ---------------------------------------------------------------- pose search
    angle_step: float = _f(30.0, "In-plane angle step of the pose search (degrees).", "pose search")
    local_angle_steps: Optional[int] = _f(None, "Search +/- this many angle steps around the current "
                                          "angle; 'none' = full 360 degrees.", "pose search")
    search_mirror: bool = _f(True, "Also search mirrored poses.", "pose search")
    shift_step: float = _f(4.0, "Translation step of the pose search (input pixels).", "pose search")
    shift_steps: int = _f(2, "Search +/- shift_steps * shift_step pixels in x and y.", "pose search")
    max_shift: Optional[float] = _f(None, "Shifts larger than this are reset to 0 after each search "
                                    "(input pixels; 'none' = box / 4).", "pose search")

    # ---------------------------------------------------------------- mask
    mask_radius: float = _f(32.0, "Radius of the soft circular mask (input pixels).", "mask")
    mask_edge_width: float = _f(8.0, "Width of the cosine soft edge of the masks (input pixels).", "mask")
    mask_encoder_input: bool = _f(True, "Mask the images fed to the encoder (not only the loss).", "mask")

    # ---------------------------------------------------------------- warm-up
    warmup_rounds: int = _f(2, "Number of initial rounds using the warm-up mask and shift limit. "
                            "The best-loss tracker is reset when warm-up ends.", "warm-up")
    warmup_mask_radius: float = _f(32.0, "Mask radius during warm-up (input pixels).", "warm-up")
    warmup_max_shift: float = _f(8.0, "Shift limit during warm-up (input pixels).", "warm-up")

    # ---------------------------------------------------------------- particle selection
    keep_fraction: float = _f(0.9, "Fraction of lowest-loss particles written to "
                              "selected_particles.npy at the end of training.", "particle selection")
    train_on_selection: bool = _f(False, "With --init-from: train only on the previous stage's "
                                  "selected particles (poses are still refined for all).",
                                  "particle selection")

    # ---------------------------------------------------------------- saving
    save_every_round: bool = _f(True, "Keep a weights+poses snapshot for every round "
                                "(checkpoints/round_NNN.pt).", "saving")

    # ---------------------------------------------------------------- helpers
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> "TrainConfig":
        if self.stage not in (1, 2, 3):
            raise ValueError(f"stage must be 1, 2 or 3 (got {self.stage})")
        if self.half_res and self.boost_decoder:
            raise ValueError("boost_decoder requires full resolution (half_res: false)")
        if self.lowpass_size % 2 != 1:
            raise ValueError("lowpass_size must be odd")
        if not 0 < self.keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0, 1]")
        if self.angle_step <= 0 or self.shift_step < 0 or self.shift_steps < 0:
            raise ValueError("angle_step must be > 0; shift_step and shift_steps must be >= 0")
        for name in ("n_rounds", "epochs_per_round", "batch_size", "latent_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        return self


# Stage presets, tuned for 128-px boxes (pixel entries are rescaled for other boxes).
STAGE_PRESETS: Dict[int, Dict[str, Any]] = {
    # Stage 1: 2x-binned images, global 360-degree + mirror search, coarse alignment.
    1: dict(half_res=True, boost_decoder=False, batch_size=128, lr=5e-4,
            angle_step=30.0, local_angle_steps=None, search_mirror=True, shift_step=4.0,
            mask_radius=32.0, mask_edge_width=8.0, warmup_mask_radius=32.0, warmup_max_shift=8.0),
    # Stage 2: full resolution, local +/-50 degree search, pixel-duplication up-sampling.
    2: dict(half_res=False, boost_decoder=False, batch_size=32, lr=2.5e-4,
            angle_step=10.0, local_angle_steps=5, search_mirror=False, shift_step=4.0,
            mask_radius=60.0, mask_edge_width=4.0, warmup_mask_radius=32.0, warmup_max_shift=4.0),
    # Stage 3: as stage 2 with a finer angle step and the learned up-sampling decoder.
    3: dict(half_res=False, boost_decoder=True, batch_size=32, lr=2e-4,
            angle_step=6.0, local_angle_steps=5, search_mirror=False, shift_step=4.0,
            mask_radius=60.0, mask_edge_width=4.0, warmup_mask_radius=32.0, warmup_max_shift=4.0),
}

_FIELD_NAMES = {f.name for f in fields(TrainConfig)}


def check_keys(values: Dict[str, Any], source: str) -> None:
    """Raise a helpful error for unknown configuration keys."""
    for key in values:
        if key not in _FIELD_NAMES:
            hint = difflib.get_close_matches(key, _FIELD_NAMES, n=1)
            hint = f" (did you mean '{hint[0]}'?)" if hint else ""
            raise KeyError(f"Unknown config key '{key}' in {source}{hint}")


def preset_for(stage: int, box_size: int = REFERENCE_BOX) -> Dict[str, Any]:
    """Stage preset with pixel-valued entries rescaled to ``box_size``."""
    if stage not in STAGE_PRESETS:
        raise ValueError(f"stage must be 1, 2 or 3 (got {stage})")
    preset = dict(STAGE_PRESETS[stage])
    scale = box_size / REFERENCE_BOX
    for key in PIXEL_FIELDS:
        if preset.get(key) is not None:
            preset[key] = float(preset[key]) * scale
    return preset


def resolve_config(stage: int, overrides: Optional[Dict[str, Any]] = None,
                   box_size: int = REFERENCE_BOX) -> TrainConfig:
    """defaults < stage preset (rescaled to box_size) < overrides."""
    overrides = dict(overrides or {})
    check_keys(overrides, "overrides")
    values = asdict(TrainConfig())
    values.update(preset_for(stage, box_size))
    values.update(overrides)
    values["stage"] = stage
    cfg = TrainConfig(**values).validate()
    for key in ("data", "out", "init_from"):
        if getattr(cfg, key):
            setattr(cfg, key, os.path.abspath(os.path.expanduser(getattr(cfg, key))))
    return cfg


# --------------------------------------------------------------------------- YAML I/O

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        values = yaml.safe_load(fh) or {}
    if not isinstance(values, dict):
        raise ValueError(f"{path}: expected a mapping of key: value pairs")
    check_keys(values, path)
    return values


def save_yaml(cfg: TrainConfig, path: str) -> None:
    with open(path, "w") as fh:
        fh.write("# cryo2dclass resolved training configuration\n")
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)


def config_to_yaml(cfg: TrainConfig) -> str:
    return yaml.safe_dump(cfg.to_dict(), sort_keys=False)


# --------------------------------------------------------------------------- argparse

def _parse_optional(kind):
    def parse(text: str):
        if text.lower() in ("none", "null", ""):
            return None
        return kind(text)
    parse.__name__ = kind.__name__
    return parse


def _show(value) -> str:
    return "none" if value is None else str(value)


def add_config_arguments(parser: argparse.ArgumentParser,
                         skip: tuple = ()) -> None:
    """Add one ``--flag`` per TrainConfig field, grouped as in the dataclass.

    Defaults are suppressed so that only flags the user actually typed end up in
    the parsed namespace (they then override preset and YAML values).
    """
    groups: Dict[str, argparse._ArgumentGroup] = {}
    for f in fields(TrainConfig):
        if f.name in skip:
            continue
        meta = f.metadata
        group = groups.setdefault(meta["group"], parser.add_argument_group(meta["group"]))
        flag = "--" + f.name.replace("_", "-")
        if f.name in STAGE_PRESETS[1]:
            values = ", ".join(f"{s}: {_show(STAGE_PRESETS[s][f.name])}" for s in STAGE_PRESETS)
            help_text = f"{meta['help']} [stage presets - {values}]"
        else:
            help_text = f"{meta['help']} [default: {_show(f.default)}]"
        kwargs: Dict[str, Any] = dict(dest=f.name, default=argparse.SUPPRESS,
                                      help=help_text.replace("%", "%%"))
        type_str = str(f.type)
        if "bool" in type_str:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            base = int if "int" in type_str else float if "float" in type_str else str
            kwargs["type"] = _parse_optional(base) if "Optional" in type_str else base
            if "choices" in meta:
                kwargs["choices"] = meta["choices"]
            kwargs["metavar"] = meta.get("metavar", base.__name__.upper())
        group.add_argument(flag, **kwargs)


def namespace_overrides(ns: argparse.Namespace) -> Dict[str, Any]:
    """Config values that were given explicitly on the command line."""
    return {k: v for k, v in vars(ns).items() if k in _FIELD_NAMES}


def replace(cfg: TrainConfig, **changes) -> TrainConfig:
    return dataclasses.replace(cfg, **changes).validate()
