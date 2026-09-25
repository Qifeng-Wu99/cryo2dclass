"""In-plane poses, image resampling and pose-search grids.

Pose convention
---------------
A pose is a 4-vector ``[angle, shift_x, shift_y, mirror]``:

* ``angle``   in-plane rotation in radians,
* ``shift_x`` translation along image columns, in pixels,
* ``shift_y`` translation along image rows, in pixels,
* ``mirror``  ``+1`` (none) or ``-1`` (mirror x before rotating).

``transform_images(img, pose)`` resamples ``img`` so that

    out(p) = img(A @ p + t),   A = R(angle) @ diag(mirror, 1)

where ``p`` are pixel coordinates relative to the image centre and ``t`` is the
shift. A particle's pose relates it to its class reference ``ref`` by
``particle = transform_images(ref, pose)``; ``align_to_reference`` inverts that.

Shifts in stored poses are in *input-image* pixels; the model converts them to
the working resolution internally (see ``model.AlignedAutoencoder``).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

ANGLE, SHIFT_X, SHIFT_Y, MIRROR = 0, 1, 2, 3
SHIFTS = slice(SHIFT_X, SHIFT_Y + 1)


def identity_poses(n: int, device=None) -> torch.Tensor:
    """(n, 4) identity poses: no rotation, no shift, no mirror."""
    poses = torch.zeros(n, 4, device=device)
    poses[:, MIRROR] = 1.0
    return poses


def sampling_matrix(angle: torch.Tensor, mirror: torch.Tensor) -> torch.Tensor:
    """A = R(angle) @ diag(mirror, 1), shape (..., 2, 2)."""
    c, s = torch.cos(angle), torch.sin(angle)
    row0 = torch.stack([c * mirror, -s], dim=-1)
    row1 = torch.stack([s * mirror, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def transform_images(images: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """Resample every image under several poses (bilinear, zero padding).

    Args:
        images: (B, C, H, W) square images.
        poses:  (B, M, 4) poses in pixels of ``images``.

    Returns:
        (B, M, C, H, W) transformed images.
    """
    B, C, H, W = images.shape
    M = poses.shape[1]
    poses = poses.to(device=images.device, dtype=images.dtype)

    ys = torch.linspace(-1, 1, H, device=images.device, dtype=images.dtype)
    xs = torch.linspace(-1, 1, W, device=images.device, dtype=images.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (H*W, 2) normalised (x, y)

    a = sampling_matrix(poses[..., ANGLE], poses[..., MIRROR])  # (B, M, 2, 2)
    t = torch.stack([2.0 * poses[..., SHIFT_X] / (W - 1), 2.0 * poses[..., SHIFT_Y] / (H - 1)], dim=-1)
    grid = base @ a.transpose(-1, -2)  # (B, M, H*W, 2): A @ p for every pixel p
    grid += t[..., None, :]

    # One grid_sample call per batch: stack the M poses along the output height.
    out = F.grid_sample(images, grid.reshape(B, M * H, W, 2),
                        mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.reshape(B, C, M, H, W).transpose(1, 2)


def compose_poses(current: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Pose update after the pose search.

    ``current`` is the particle's pose before the search, ``delta`` the best
    search candidate (angle/mirror applied to the reconstruction, shift added
    to the current shift). Both use the same pixel units.

    Derivation: the search matches ``I(A(-a*m, m) p)`` with ``O(A(da, dm) p + t)``;
    rewriting as ``I(A(-a'*m', m') q) = O(q + t)`` gives
    ``a' = dm * a + da``, ``m' = m * dm``, ``t' = t``.
    """
    new = torch.empty_like(current)
    new[:, ANGLE] = torch.remainder(delta[:, MIRROR] * current[:, ANGLE] + delta[:, ANGLE], 2 * math.pi)
    new[:, SHIFTS] = current[:, SHIFTS] + delta[:, SHIFTS]
    new[:, MIRROR] = current[:, MIRROR] * delta[:, MIRROR]
    return new


def undo_rotation(poses: torch.Tensor) -> torch.Tensor:
    """Poses that rotate/mirror a particle back to the reference orientation (no shift)."""
    inv = torch.zeros_like(poses)
    inv[:, ANGLE] = -poses[:, ANGLE] * poses[:, MIRROR]
    inv[:, MIRROR] = poses[:, MIRROR]
    return inv


def align_to_reference(images: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """Bring particles into the reference frame with a single resampling.

    Inverse of ``transform_images(ref, pose)``: returns (B, C, H, W) images
    ``ref_estimate(q) = particle(A^-1 q - A^-1 t)``.
    """
    inv = undo_rotation(poses)
    a_inv = sampling_matrix(inv[:, ANGLE], inv[:, MIRROR])  # (B, 2, 2)
    inv[:, SHIFTS] = -(a_inv @ poses[:, SHIFTS, None])[..., 0]
    return transform_images(images, inv[:, None])[:, 0]


def pose_change(old: torch.Tensor, new: torch.Tensor) -> dict:
    """Summary of how much the poses moved during one search (for logging)."""
    dangle = torch.rad2deg(new[:, ANGLE] - old[:, ANGLE]) % 360
    dangle = torch.minimum(dangle, 360 - dangle)
    dshift = torch.linalg.norm(new[:, SHIFTS] - old[:, SHIFTS], dim=1)
    flipped = (new[:, MIRROR] != old[:, MIRROR]).float()
    return {
        "angle_change_deg": float(dangle.mean()),
        "shift_change_px": float(dshift.mean()),
        "mirror_flip_fraction": float(flipped.mean()),
    }


def build_pose_grids(angle_step: float, local_angle_steps: Optional[int], mirror: bool,
                     shift_step: float, shift_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Candidate poses used by the model.

    Returns:
        views:  (V, 4) rotations/mirrors fed to the encoder. ``views[0]`` is the
                identity; the encoder max-pools over all views.
        search: (T, 4) = every view combined with every shift on a
                (2*shift_steps + 1)^2 grid, sorted from small to large moves.
    All shifts are in pixels of the working resolution.
    """
    if local_angle_steps is None:
        angles = torch.arange(0.0, 360.0, angle_step)
    else:
        angles = torch.arange(-local_angle_steps, local_angle_steps + 1, dtype=torch.float32) * angle_step
    angles = angles[torch.argsort(angles.abs(), stable=True)]

    views = torch.zeros(len(angles), 4)
    views[:, ANGLE] = torch.deg2rad(angles)
    views[:, MIRROR] = 1.0
    if mirror:
        mirrored = views.clone()
        mirrored[:, MIRROR] = -1.0
        views = torch.cat([views, mirrored])

    offsets = torch.arange(-shift_steps, shift_steps + 1, dtype=torch.float32) * shift_step
    sy, sx = torch.meshgrid(offsets, offsets, indexing="ij")
    shifts = torch.stack([sx.reshape(-1), sy.reshape(-1)], dim=1)  # (S, 2)

    search = views[None].repeat(len(shifts), 1, 1)  # (S, V, 4)
    search[..., SHIFTS] += shifts[:, None, :]
    search = search.reshape(-1, 4)
    search = search[torch.argsort(torch.linalg.norm(search, dim=1), stable=True)]
    return views, search
