"""The full model: encoder + decoder with the pose search built into the loss.

For a batch of particles with current poses the forward pass

1. rotates/mirrors each particle back to the reference orientation and makes
   V views of it (the encoder input) - the identity view is the *target*;
2. encodes the views into a latent vector (+ intensity scale/offset);
3. decodes the latent vector into a reconstruction in the reference frame;
4. moves the reconstruction under every candidate pose of the search grid and
   compares each candidate with the (masked, RMS-normalised) target.

The loss keeps only the best candidate per particle, so network training and
pose assignment are solved together. After every round ``search`` re-assigns
the poses of all particles from the best candidates.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainConfig
from .filters import gaussian_kernel, lowpass, soft_circular_mask
from .networks import BoostDecoder, Decoder, Encoder
from .transforms import ANGLE, MIRROR, SHIFTS, build_pose_grids, compose_poses, transform_images, undo_rotation


class ForwardOutput(NamedTuple):
    target: torch.Tensor     # (B, h, w)    particle in reference orientation, masked, RMS = 1
    latent: torch.Tensor     # (B, D)       embedding
    recon: torch.Tensor      # (B, 1, h, w) reconstruction in the reference frame
    residuals: torch.Tensor  # (B, T, h, w) target - candidate reconstruction, per search pose


class SearchResult(NamedTuple):
    latent: torch.Tensor     # (B, D)
    loss: torch.Tensor       # (B,)   MSE of the best candidate
    poses: torch.Tensor      # (B, 4) updated poses (input pixels)


class AlignedAutoencoder(nn.Module):
    """Rotation-invariant autoencoder with an exhaustive in-plane pose search.

    Args:
        box_size: input box size (pixels); must be divisible by 16.
        cfg: training configuration (model, mask and pose-search settings).
    """

    def __init__(self, box_size: int, cfg: TrainConfig):
        super().__init__()
        if box_size % 16:
            raise ValueError(f"box size must be divisible by 16, got {box_size}")
        self.box_size = box_size
        self.half_res = cfg.half_res
        self.scale = 2 if cfg.half_res else 1           # input pixels per working pixel
        self.work_size = box_size // self.scale
        self.latent_noise = cfg.latent_noise
        self.latent_norm_max = cfg.latent_norm_max
        self.latent_reg_weight = cfg.latent_reg_weight
        self.mask_encoder_input = cfg.mask_encoder_input

        self.encoder = Encoder(cfg.latent_dim, cfg.latent_dropout)
        self.decoder = Decoder(box_size // 2, cfg.latent_dim)
        self.boost = BoostDecoder() if cfg.boost_decoder else None

        s = self.scale
        views, search = build_pose_grids(cfg.angle_step, cfg.local_angle_steps, cfg.search_mirror,
                                         cfg.shift_step / s, cfg.shift_steps)
        buffers = {
            "lowpass_kernel": gaussian_kernel(cfg.lowpass_size),
            "mask": soft_circular_mask(self.work_size, cfg.mask_radius / s, cfg.mask_edge_width / s),
            "warmup_mask": soft_circular_mask(self.work_size, cfg.warmup_mask_radius / s,
                                              cfg.mask_edge_width / s),
            "view_poses": views,      # (V, 4)
            "search_poses": search,   # (T, 4), working pixels
        }
        for name, value in buffers.items():  # derived from the config -> not saved in checkpoints
            self.register_buffer(name, value, persistent=False)

    @classmethod
    def from_config(cls, cfg: TrainConfig, box_size: int) -> "AlignedAutoencoder":
        return cls(box_size, cfg)

    # ------------------------------------------------------------------ units
    def to_working(self, poses: torch.Tensor) -> torch.Tensor:
        poses = poses.clone()
        poses[:, SHIFTS] /= self.scale
        return poses

    def to_input(self, poses: torch.Tensor) -> torch.Tensor:
        poses = poses.clone()
        poses[:, SHIFTS] *= self.scale
        return poses

    # ------------------------------------------------------------------ pieces
    def _lowpass(self, x: torch.Tensor) -> torch.Tensor:
        return lowpass(x, self.lowpass_kernel)

    def decode(self, latent: torch.Tensor, intensity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Latent (B, D) -> reconstruction (B, 1, h, w) at the working resolution."""
        x = self.decoder(latent)
        if intensity is not None:
            x = x * intensity[:, 0, None, None, None] + intensity[:, 1, None, None, None]
        x = self._lowpass(x)
        if not self.half_res:
            x = self.boost(x) if self.boost is not None else F.interpolate(x, scale_factor=2, mode="nearest")
            x = self._lowpass(x)
        return x

    def make_views(self, images: torch.Tensor, poses: torch.Tensor, mask: torch.Tensor):
        """Encoder input (B, V, box/2, box/2) and target (B, h, w) for particles at ``poses``."""
        B = images.shape[0]
        view_poses = self.view_poses.expand(B, -1, -1).clone()
        undo = undo_rotation(poses)
        view_poses[..., ANGLE] += undo[:, None, ANGLE]
        view_poses[..., MIRROR] *= undo[:, None, MIRROR]

        x = images[:, None]
        if self.half_res:
            x = self._lowpass(F.avg_pool2d(x, 2))
            views = transform_images(x, view_poses)[:, :, 0]
            masked = views * mask
            encoder_in = masked if self.mask_encoder_input else views
        else:
            views = transform_images(self._lowpass(x), view_poses)[:, :, 0]
            masked = views * mask
            encoder_in = F.avg_pool2d(masked if self.mask_encoder_input else views, 2)
        return encoder_in, masked[:, 0]  # view 0 is the identity

    # ------------------------------------------------------------------ forward / loss
    def forward(self, images: torch.Tensor, poses: torch.Tensor, warmup: bool = False) -> ForwardOutput:
        """
        Args:
            images: (B, H, W) particles at input resolution.
            poses:  (B, 4) current poses, shifts in input pixels.
            warmup: use the warm-up mask.
        """
        mask = self.warmup_mask if warmup else self.mask
        poses = self.to_working(poses)
        encoder_in, target = self.make_views(images, poses, mask)

        latent, intensity = self.encoder(encoder_in)
        if self.training and self.latent_noise > 0:
            latent = latent + torch.randn_like(latent) * self.latent_noise
        recon = self.decode(latent, intensity)

        B = images.shape[0]
        candidates = self.search_poses.expand(B, -1, -1).clone()
        candidates[..., SHIFTS] += poses[:, None, SHIFTS]
        moved = transform_images(recon, candidates)[:, :, 0] * mask       # (B, T, h, w)

        rms = target.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
        target = target / rms
        return ForwardOutput(target, latent, recon, target[:, None] - moved)

    def loss(self, out: ForwardOutput) -> torch.Tensor:
        """Best-candidate MSE averaged over the batch + latent-norm penalty."""
        best = out.residuals.pow(2).mean(dim=(-2, -1)).min(dim=1).values
        penalty = F.relu(out.latent.pow(2).sum(dim=-1) - self.latent_norm_max ** 2)
        return best.mean() + self.latent_reg_weight * penalty.mean()

    @torch.no_grad()
    def search(self, images: torch.Tensor, poses: torch.Tensor, warmup: bool = False) -> SearchResult:
        """Embed particles and re-assign their poses from the best search candidate (eval mode)."""
        was_training = self.training
        self.eval()
        out = self.forward(images, poses, warmup)
        self.train(was_training)
        mse = out.residuals.pow(2).mean(dim=(-2, -1))
        best_mse, best = mse.min(dim=1)
        new = compose_poses(self.to_working(poses), self.search_poses[best])
        return SearchResult(out.latent, best_mse, self.to_input(new))

    def limit_shifts(self, poses: torch.Tensor, max_shift: float) -> torch.Tensor:
        """Reset shifts beyond ``max_shift`` (input px) and round to whole working pixels."""
        poses = self.to_working(poses)
        shifts = poses[:, SHIFTS]
        shifts[torch.linalg.norm(shifts, dim=1) > max_shift / self.scale] = 0.0
        poses[:, SHIFTS] = torch.round(shifts)
        return self.to_input(poses)

    # ------------------------------------------------------------------ weights
    def load_weights(self, state: dict, allow_missing_boost: bool = False) -> None:
        """Load a state dict; optionally allow the boost decoder to be missing (stage 2 -> 3)."""
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if not (allow_missing_boost and k.startswith("boost."))]
        unexpected = [k for k in unexpected if not k.startswith("boost.")]
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint does not match the model. Missing: {missing}; "
                               f"unexpected: {unexpected}")
