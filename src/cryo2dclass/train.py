"""Training loop (single GPU, multi-GPU via ``accelerate launch``, or CPU).

Each round:
  1. ``epochs_per_round`` epochs of gradient steps on the training particles,
     with the pose search inside the loss;
  2. a pose search over *all* particles, which re-assigns every particle's pose
     (and yields its embedding and reconstruction error);
  3. checkpoints and metrics are written to the run directory.

With several GPUs each process trains on a disjoint shard of every epoch and
the pose search is split across processes and gathered, so all processes keep
identical poses.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from . import checkpoint as ck
from .config import TrainConfig, save_yaml
from .data import load_particles
from .model import AlignedAutoencoder, SearchResult
from .transforms import identity_poses, pose_change

log = logging.getLogger("cryo2dclass")


def shard_batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator,
                  rank: int = 0, world: int = 1) -> List[np.ndarray]:
    """Shuffle ``indices`` and return this process's mini-batches.

    All processes draw the same permutation; process ``rank`` takes every
    ``world``-th element (padded by wrap-around so all get the same number of
    batches, as DDP requires).
    """
    perm = rng.permutation(indices)
    if world > 1:
        perm = np.resize(perm, math.ceil(len(perm) / world) * world)[rank::world]
    return [perm[i:i + batch_size] for i in range(0, len(perm), batch_size)]


def search_all(model: AlignedAutoencoder, images: torch.Tensor, poses: torch.Tensor, warmup: bool,
               batch_size: int, accelerator: Accelerator, desc: str = "pose search") -> SearchResult:
    """Run ``model.search`` over every particle (split across processes); results on CPU."""
    n = len(images)
    world, rank = accelerator.num_processes, accelerator.process_index
    per_rank = math.ceil(n / world)
    idx = np.resize(np.arange(n), per_rank * world)[rank * per_rank:(rank + 1) * per_rank]
    parts = []
    for start in tqdm(range(0, len(idx), batch_size), desc=desc, leave=False,
                      disable=None if accelerator.is_main_process else True):
        b = torch.from_numpy(idx[start:start + batch_size])
        parts.append(model.search(images[b].to(accelerator.device), poses[b].to(accelerator.device), warmup))
    fields = [torch.cat([getattr(p, f) for p in parts]) for f in SearchResult._fields]
    if world > 1:
        fields = [accelerator.gather(t) for t in fields]
    return SearchResult(*(t[:n].cpu() for t in fields))


def _attach_log_file(path: str) -> logging.Handler:
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(handler)
    return handler


def train(cfg: TrainConfig, resume: bool = False, accelerator: Optional[Accelerator] = None) -> str:
    """Train one stage. Returns the run directory.

    Inputs:  ``cfg.data`` (particle stack) and optionally ``cfg.init_from`` (previous stage).
    Outputs: the run directory ``cfg.out`` (layout in :mod:`cryo2dclass.checkpoint`).
    """
    if not cfg.data or not cfg.out:
        raise ValueError("Both 'data' and 'out' must be set")
    accelerator = accelerator or Accelerator()
    is_main = accelerator.is_main_process
    run_dir = cfg.out
    handler = None
    if is_main:
        os.makedirs(os.path.join(run_dir, ck.CKPT_DIR), exist_ok=True)
        save_yaml(cfg, os.path.join(run_dir, ck.CONFIG_FILE))
        handler = _attach_log_file(os.path.join(run_dir, ck.LOG_FILE))
    accelerator.wait_for_everyone()

    try:
        return _train(cfg, resume, accelerator, run_dir, is_main)
    finally:
        if handler is not None:
            log.removeHandler(handler)
            handler.close()


def _train(cfg: TrainConfig, resume: bool, accelerator: Accelerator, run_dir: str, is_main: bool) -> str:
    set_seed(cfg.seed, device_specific=True)       # dropout / latent noise differ per process
    rng = np.random.default_rng(cfg.seed)            # shuffling: identical on every process
    torch.backends.cudnn.benchmark = True
    device = accelerator.device

    # ---------------------------------------------------------------- data + model
    images = torch.from_numpy(load_particles(cfg.data, cfg.normalize))
    n, box = images.shape[0], images.shape[-1]
    model = AlignedAutoencoder(box, cfg)
    poses = identity_poses(n)
    train_idx = np.arange(n)
    state = {"round": 0, "best_loss": math.inf, "history": {"rounds": [], "batch_losses": []}}
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    log.info("Stage %d | %d particles, box %d px | %s resolution%s | %d processes on %s",
             cfg.stage, n, box, "half" if cfg.half_res else "full",
             " + boost decoder" if cfg.boost_decoder else "", accelerator.num_processes, device)

    if resume:
        ckpt = ck.load_checkpoint(ck.checkpoint_path(run_dir, "last"))
        _check_compatible(ckpt, n, box, run_dir)
        model.load_weights(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for group in optimizer.param_groups:  # a changed --lr applies on resume
            group["lr"] = cfg.lr
        poses = ckpt["poses"]
        state = ckpt["train_state"]
        train_idx = state.pop("train_idx").numpy()
        rng.bit_generator.state = state.pop("rng")
        log.info("Resuming %s after round %d", run_dir, state["round"])
    elif cfg.init_from:
        ckpt = ck.load_checkpoint(ck.checkpoint_path(cfg.init_from, cfg.init_checkpoint))
        _check_compatible(ckpt, n, box, cfg.init_from)
        model.load_weights(ckpt["model"], allow_missing_boost=True)
        poses = ckpt["poses"]
        log.info("Initialised from %s (%s checkpoint, stage %d round %d)", cfg.init_from,
                 cfg.init_checkpoint, ckpt["stage"], ckpt["round"])
        if cfg.train_on_selection:
            train_idx = np.load(os.path.join(cfg.init_from, ck.SELECTION_FILE))
            log.info("Training on %d selected particles from the previous stage", len(train_idx))
    elif cfg.stage > 1:
        log.warning("Stage %d without --init-from: training from scratch with identity poses", cfg.stage)

    model, optimizer = accelerator.prepare(model, optimizer)
    core: AlignedAutoencoder = accelerator.unwrap_model(model)
    max_shift = cfg.max_shift if cfg.max_shift is not None else box / 4
    world, rank = accelerator.num_processes, accelerator.process_index

    if state["round"] >= cfg.n_rounds:
        log.info("Nothing to do: %d/%d rounds already completed (raise --n-rounds to continue)",
                 state["round"], cfg.n_rounds)

    # ---------------------------------------------------------------- rounds
    for r in range(state["round"], cfg.n_rounds):
        t0 = time.time()
        warmup = r < cfg.warmup_rounds
        model.train()
        losses: List[float] = []
        for epoch in range(cfg.epochs_per_round):
            batches = shard_batches(train_idx, cfg.batch_size, rng, rank, world)
            bar = tqdm(batches, leave=False, disable=None if is_main else True,
                       desc=f"round {r + 1}/{cfg.n_rounds} epoch {epoch + 1}/{cfg.epochs_per_round}")
            for idx in bar:
                idx = torch.from_numpy(idx)
                x, p = images[idx].to(device, non_blocking=True), poses[idx].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = core.loss(model(x, p, warmup=warmup))
                accelerator.backward(loss)
                optimizer.step()
                losses.append(loss.item())
                bar.set_postfix(loss=f"{losses[-1]:.4f}")
        train_loss = accelerator.reduce(torch.tensor(float(np.mean(losses)), device=device), "mean").item()

        # ---- pose search over all particles
        result = search_all(core, images, poses, warmup, cfg.batch_size, accelerator)
        new_poses = core.limit_shifts(result.poses, cfg.warmup_max_shift if warmup else max_shift)
        change = pose_change(poses, new_poses)
        poses = new_poses

        # ---- bookkeeping (best-loss tracking restarts once the warm-up mask is dropped)
        if r == cfg.warmup_rounds:
            state["best_loss"] = math.inf
        improved = train_loss < state["best_loss"]
        if improved:
            state["best_loss"] = train_loss
        record = {"round": r, "warmup": warmup, "train_loss": train_loss,
                  "search_loss": float(result.loss.mean()), **change,
                  "best": improved, "seconds": round(time.time() - t0, 1)}
        state["history"]["rounds"].append(record)
        state["history"]["batch_losses"].extend(losses)
        state["round"] = r + 1

        if is_main:
            _save_round(run_dir, core, poses, result, r, cfg, optimizer, state, train_idx, rng, improved)
            log.info("round %d/%d%s | train loss %.5f | search loss %.5f | dAngle %.2f deg  dShift %.2f px"
                     "  mirror flips %.1f%% | %.0f s%s", r + 1, cfg.n_rounds, " (warm-up)" if warmup else "",
                     train_loss, record["search_loss"], change["angle_change_deg"], change["shift_change_px"],
                     100 * change["mirror_flip_fraction"], record["seconds"], " | new best" if improved else "")
        accelerator.wait_for_everyone()

    if is_main:
        log.info("Done. Results in %s", run_dir)
    return run_dir


def _check_compatible(ckpt: dict, n: int, box: int, source: str) -> None:
    if ckpt["poses"].shape[0] != n or ckpt["box_size"] != box:
        raise ValueError(f"{source} was trained on {ckpt['poses'].shape[0]} particles of box "
                         f"{ckpt['box_size']} px, but the current stack has {n} particles of box {box} px. "
                         "Use the same particle stack for all stages.")


def _save_round(run_dir, model, poses, result, r, cfg, optimizer, state, train_idx, rng, improved):
    ckpt_dir = os.path.join(run_dir, ck.CKPT_DIR)
    train_state = {**state, "train_idx": torch.from_numpy(np.asarray(train_idx)),
                   "rng": rng.bit_generator.state}
    ck.save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, poses, r, cfg, optimizer, train_state)
    if improved:
        ck.save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model, poses, r, cfg)
    if cfg.save_every_round:
        ck.save_checkpoint(os.path.join(ckpt_dir, f"round_{r:03d}.pt"), model, poses, r, cfg)

    # Always-current per-particle results of the latest pose search.
    loss = result.loss.numpy()
    np.save(os.path.join(run_dir, ck.EMBEDDINGS_FILE), result.latent.numpy())
    np.save(os.path.join(run_dir, ck.LOSS_FILE), loss)
    keep = np.sort(np.argsort(loss)[:max(1, int(len(loss) * cfg.keep_fraction))])
    np.save(os.path.join(run_dir, ck.SELECTION_FILE), keep)
    ck.save_poses_csv(os.path.join(run_dir, ck.POSES_FILE), poses, {"loss": loss})
    ck.write_history(run_dir, state["history"])
