"""Fixed (non-learned) image filters: soft circular mask and Gaussian low-pass."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def soft_circular_mask(size: int, radius: float, edge_width: float) -> torch.Tensor:
    """(size, size) mask: 1 inside ``radius``, cosine fall-off to 0 over ``edge_width`` pixels.

    Equivalent to EMAN2's ``mask.soft`` with ``outer_radius = radius - size / 2``.
    """
    y, x = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask = np.ones((size, size), dtype=np.float32)
    if edge_width > 0:
        edge = (r > radius) & (r <= radius + edge_width)
        mask[edge] = 0.5 + 0.5 * np.cos(np.pi * (r[edge] - radius) / edge_width)
    mask[r > radius + edge_width] = 0.0
    return torch.from_numpy(mask)


def gaussian_kernel(size: int) -> torch.Tensor:
    """(1, 1, size, size) normalised kernel exp(-r^2 / 4)."""
    half = size // 2
    ax = torch.arange(-half, half + 1, dtype=torch.float32)
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(gx ** 2 + gy ** 2) / 4)
    return (k / k.sum()).view(1, 1, size, size)


def lowpass(images: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Low-pass filter single-channel images (B, 1, H, W) with a same-size output."""
    return F.conv2d(images, kernel, padding=kernel.shape[-1] // 2)
