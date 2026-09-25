"""Reading and writing particle stacks and label files.

Particle stacks
    ``.npy``                     float array of shape (N, H, W)
    ``.mrcs .mrc .st .ali .map`` MRC stack (N sections of H x W), e.g. exported
                                 from RELION, cryoSPARC, EMAN2 or IMOD

Images must be square with a box size divisible by 16 (e.g. 64, 96, 128, 256).
Particles are expected to be roughly centred; background normalisation is
recommended (or pass ``--normalize``).

Label files (optional, only for measuring accuracy)
    ``.npy``, or ``.txt`` / ``.csv`` with one integer class label per line
    (a header line is allowed for .csv).
"""

from __future__ import annotations

import os
from typing import Sequence, Tuple

import numpy as np

MRC_EXTENSIONS = (".mrcs", ".mrc", ".st", ".ali", ".map")
NPY_EXTENSIONS = (".npy",)


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _open_raw(path: str):
    """Memory-mapped view of the stack; returns (array_like, closer)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Particle stack not found: {path}")
    ext = _ext(path)
    if ext in NPY_EXTENSIONS:
        return np.load(path, mmap_mode="r"), (lambda: None)
    if ext in MRC_EXTENSIONS:
        import mrcfile

        mrc = mrcfile.mmap(path, mode="r", permissive=True)
        return mrc.data, mrc.close
    raise ValueError(f"Unsupported particle file '{path}'. "
                     f"Use one of: {', '.join(NPY_EXTENSIONS + MRC_EXTENSIONS)}")


def _check_shape(shape: Tuple[int, ...], path: str) -> Tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError(f"{path}: expected a stack of 2-D images (N, H, W), got shape {shape}")
    n, h, w = shape
    if h != w:
        raise ValueError(f"{path}: images must be square, got {h} x {w}")
    if h % 16:
        raise ValueError(f"{path}: box size must be divisible by 16, got {h}. "
                         "Crop or pad the particles (e.g. to 128 px).")
    return int(n), int(h), int(w)


def stack_shape(path: str) -> Tuple[int, int, int]:
    """(N, H, W) of a particle stack without loading it into memory."""
    data, close = _open_raw(path)
    try:
        return _check_shape(tuple(data.shape), path)
    finally:
        close()


def load_particles(path: str, normalize: bool = False) -> np.ndarray:
    """Load a particle stack as a contiguous float32 array of shape (N, H, W)."""
    data, close = _open_raw(path)
    try:
        _check_shape(tuple(data.shape), path)
        images = np.array(data, dtype=np.float32)  # copy out of the memory map
    finally:
        close()
    if not np.isfinite(images).all():
        raise ValueError(f"{path}: contains NaN or Inf values")
    if normalize:
        mean = images.mean(axis=(1, 2), keepdims=True)
        std = images.std(axis=(1, 2), keepdims=True)
        images = (images - mean) / np.maximum(std, 1e-8)
    return images


def read_stack(path: str) -> np.ndarray:
    """Read any image stack written by :func:`save_stack` (no shape checks)."""
    data, close = _open_raw(path)
    try:
        return np.array(data, dtype=np.float32)
    finally:
        close()


def save_stack(path: str, images: np.ndarray) -> None:
    """Save an image stack (N, H, W) as .mrcs/.mrc or .npy (chosen by extension)."""
    images = np.asarray(images, dtype=np.float32)
    ext = _ext(path)
    if ext in MRC_EXTENSIONS:
        import mrcfile

        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(images)
    elif ext in NPY_EXTENSIONS:
        np.save(path, images)
    else:
        raise ValueError(f"Unsupported output extension for {path}")


def load_labels(path: str) -> np.ndarray:
    """Integer class labels, one per particle."""
    ext = _ext(path)
    if ext == ".npy":
        labels = np.load(path)
    elif ext in (".txt", ".csv", ".tsv", ""):
        delimiter = "," if ext == ".csv" else None
        with open(path) as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        if lines and not lines[0].split(delimiter)[0].lstrip("-").isdigit():
            lines = lines[1:]  # header
        labels = np.array([int(float(ln.split(delimiter)[-1])) for ln in lines])
    else:
        raise ValueError(f"Unsupported label file '{path}' (use .npy, .txt or .csv)")
    labels = np.asarray(labels).reshape(-1)
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int64)
    return labels


def labels_from_counts(counts: Sequence[int]) -> np.ndarray:
    """Labels for a stack sorted by class, e.g. counts=(5000, 5000) -> 0...0 1...1."""
    return np.repeat(np.arange(len(counts)), counts)
