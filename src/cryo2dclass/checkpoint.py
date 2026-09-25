"""Run-directory layout and checkpoint I/O.

A run directory produced by ``cryo2dclass train``::

    <run>/
    ├── config.yaml               resolved configuration (read by evaluate/visualize)
    ├── train.log                 console log
    ├── history.json              per-round metrics + per-batch losses
    ├── checkpoints/
    │   ├── best.pt               lowest mean training loss (after warm-up)
    │   ├── last.pt               most recent round, incl. optimizer (for --resume)
    │   └── round_NNN.pt          per-round snapshots (if save_every_round)
    ├── poses.csv                 final per-particle poses (angle, shifts in input px, mirror)
    ├── embeddings.npy            (N, latent_dim) latent vectors from the final pose search
    ├── particle_loss.npy         (N,) per-particle reconstruction error
    └── selected_particles.npy    indices of the keep_fraction lowest-loss particles

Every checkpoint holds the model weights *and* the particle poses those
weights produced, so any checkpoint can seed the next stage or an evaluation.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .config import TrainConfig, load_yaml
from .model import AlignedAutoencoder
from .transforms import ANGLE, MIRROR, SHIFT_X, SHIFT_Y

FORMAT_VERSION = 1
CONFIG_FILE = "config.yaml"
HISTORY_FILE = "history.json"
LOG_FILE = "train.log"
CKPT_DIR = "checkpoints"
POSES_FILE = "poses.csv"
EMBEDDINGS_FILE = "embeddings.npy"
LOSS_FILE = "particle_loss.npy"
SELECTION_FILE = "selected_particles.npy"


def checkpoint_path(run_dir: str, name: str = "best") -> str:
    """Path of checkpoint ``name`` (best, last, round_NNN or a round number)."""
    if str(name).isdigit():
        name = f"round_{int(name):03d}"
    path = os.path.join(run_dir, CKPT_DIR, f"{name}.pt")
    if not os.path.exists(path):
        available = sorted(os.listdir(os.path.join(run_dir, CKPT_DIR))) \
            if os.path.isdir(os.path.join(run_dir, CKPT_DIR)) else []
        raise FileNotFoundError(f"No checkpoint '{name}' in {run_dir}. Available: {available}")
    return path


def save_checkpoint(path: str, model: AlignedAutoencoder, poses: torch.Tensor, round_idx: int,
                    cfg: TrainConfig, optimizer: Optional[torch.optim.Optimizer] = None,
                    train_state: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "stage": cfg.stage,
        "round": round_idx,
        "box_size": model.box_size,
        "config": cfg.to_dict(),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "poses": poses.detach().cpu(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if train_state is not None:
        payload["train_state"] = train_state
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: an interrupted save never corrupts the old file


def load_checkpoint(path: str, map_location="cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if ckpt.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported checkpoint format {ckpt.get('format_version')}")
    return ckpt


def load_run_config(run_dir: str) -> TrainConfig:
    path = os.path.join(run_dir, CONFIG_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{run_dir} is not a cryo2dclass run directory (no {CONFIG_FILE})")
    return TrainConfig(**load_yaml(path))


def load_model(run_dir: str, checkpoint: str = "best", device="cpu"
               ) -> Tuple[TrainConfig, AlignedAutoencoder, torch.Tensor, Dict[str, Any]]:
    """(config, model in eval mode, poses (N, 4), raw checkpoint) of a trained run."""
    cfg = load_run_config(run_dir)
    ckpt = load_checkpoint(checkpoint_path(run_dir, checkpoint))
    model = AlignedAutoencoder(ckpt["box_size"], cfg)
    model.load_weights(ckpt["model"])
    model.to(device).eval()
    return cfg, model, ckpt["poses"], ckpt


def save_poses_csv(path: str, poses: torch.Tensor, extra: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Human-readable per-particle table (one row per particle, input-pixel units)."""
    poses = poses.detach().cpu().numpy()
    extra = extra or {}
    header = ["particle", "angle_deg", "shift_x_px", "shift_y_px", "mirror"] + list(extra)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, p in enumerate(poses):
            row = [i, f"{np.rad2deg(p[ANGLE]):.3f}", f"{p[SHIFT_X]:.2f}", f"{p[SHIFT_Y]:.2f}", int(p[MIRROR])]
            row += [_fmt(extra[k][i]) for k in extra]
            writer.writerow(row)


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    return int(v) if isinstance(v, (np.integer,)) else v


def read_history(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, HISTORY_FILE)
    if not os.path.exists(path):
        return {"rounds": [], "batch_losses": []}
    with open(path) as fh:
        return json.load(fh)


def write_history(run_dir: str, history: Dict[str, Any]) -> None:
    tmp = os.path.join(run_dir, HISTORY_FILE + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(history, fh)
    os.replace(tmp, os.path.join(run_dir, HISTORY_FILE))
