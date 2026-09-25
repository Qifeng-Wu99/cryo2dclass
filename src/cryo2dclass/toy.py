"""Synthetic particle stacks for testing an installation.

Each class is a different arrangement of Gaussian blobs (no rotational or
mirror symmetry). Particles get a random in-plane rotation, mirror and small
shift plus Gaussian noise; true labels and poses are saved alongside.

    python -m cryo2dclass.toy --out toy_data --n 2000 --box 64 --classes 2 --snr 1.0
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np
import torch

from .data import save_stack
from .transforms import transform_images


def class_templates(n_classes: int, box: int, seed: int = 0) -> np.ndarray:
    """(K, box, box) templates: a shared asymmetric core plus class-specific blobs."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:box, :box].astype(np.float32)
    c, u = (box - 1) / 2, box / 80  # u: size unit (particle fills ~60% of the box)
    sigma2 = (3.0 * u) ** 2 * 2

    def blob(dx, dy, amp=1.0):
        return amp * np.exp(-((x - c - dx * u) ** 2 + (y - c - dy * u) ** 2) / sigma2)

    core = blob(0, 0) + blob(8, 0, 0.9) + blob(16, 0, 0.8) + blob(0, 8, 0.7)
    templates = []
    for _ in range(n_classes):
        extra = np.zeros_like(core)
        for _ in range(2):
            r, a = rng.uniform(6, 14), rng.uniform(0, 2 * math.pi)
            extra += blob(r * math.cos(a), r * math.sin(a), rng.uniform(0.6, 1.0))
        templates.append(core + extra)
    return np.stack(templates).astype(np.float32)


def make_toy_dataset(out_dir: str, n: int = 2000, box: int = 64, n_classes: int = 2,
                     snr: float = 1.0, max_shift: float = 2.0, seed: int = 0) -> str:
    """Write particles.mrcs, labels.txt, true_poses.csv and templates.mrcs to ``out_dir``."""
    rng = np.random.default_rng(seed)
    templates = class_templates(n_classes, box, seed)
    labels = np.sort(rng.integers(0, n_classes, n))  # sorted by class, like many benchmark stacks
    poses = np.stack([rng.uniform(0, 2 * math.pi, n), rng.uniform(-max_shift, max_shift, n),
                      rng.uniform(-max_shift, max_shift, n), rng.choice([-1.0, 1.0], n)], 1).astype(np.float32)
    clean = transform_images(torch.from_numpy(templates[labels])[:, None],
                             torch.from_numpy(poses)[:, None])[:, 0, 0].numpy()
    noise_std = clean.std() / math.sqrt(snr)
    particles = clean + rng.normal(0, noise_std, clean.shape).astype(np.float32)
    particles = (particles - particles.mean(axis=(1, 2), keepdims=True)) / particles.std(axis=(1, 2), keepdims=True)

    os.makedirs(out_dir, exist_ok=True)
    save_stack(os.path.join(out_dir, "particles.mrcs"), particles)
    save_stack(os.path.join(out_dir, "templates.mrcs"), templates)
    np.savetxt(os.path.join(out_dir, "labels.txt"), labels, fmt="%d")
    with open(os.path.join(out_dir, "true_poses.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["particle", "label", "angle_deg", "shift_x_px", "shift_y_px", "mirror"])
        for i, (p, lab) in enumerate(zip(poses, labels)):
            w.writerow([i, lab, f"{math.degrees(p[0]):.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}", int(p[3])])
    return os.path.join(out_dir, "particles.mrcs")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic particle stack.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--n", type=int, default=2000, help="Number of particles (default: 2000).")
    p.add_argument("--box", type=int, default=64, help="Box size, multiple of 16 (default: 64).")
    p.add_argument("--classes", type=int, default=2, help="Number of classes (default: 2).")
    p.add_argument("--snr", type=float, default=1.0, help="Signal-to-noise ratio (variance; default: 1).")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    path = make_toy_dataset(a.out, a.n, a.box, a.classes, a.snr, seed=a.seed)
    print(f"wrote {path} (+ labels.txt, true_poses.csv, templates.mrcs)")


if __name__ == "__main__":
    main()
