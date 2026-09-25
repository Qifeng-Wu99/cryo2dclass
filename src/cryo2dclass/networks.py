"""Neural network components.

Shapes use B = batch, V = views (rotated / mirrored copies of each particle),
h x w = half of the input box (the encoder always sees 2x-binned images and the
decoder always produces half-resolution images).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Amplitude of the sine output activations (reconstructions lie in [-10, 10]).
OUTPUT_AMPLITUDE = 10.0


class Encoder(nn.Module):
    """Rotation-pooled CNN encoder.

    Each view is processed by two conv layers, then features are max-pooled
    across views, which makes the embedding (approximately) invariant to the
    rotations/mirrors in the view set.

    Input:  views (B, V, h, w)
    Output: latent (B, latent_dim) and intensity (B, 2) = per-particle
            (scale, offset) applied to the decoded image.
    """

    def __init__(self, latent_dim: int = 8, latent_dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.feature_dropout = nn.Dropout2d(0.1)
        self.pooled_dropout = nn.Dropout(0.1)
        self.latent_dropout = nn.Dropout(latent_dropout)
        # intensity head: (scale, offset), initialised around (1, 0)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 2)
        # latent head
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, latent_dim)
        self.register_buffer("intensity_bias", torch.tensor([1.0, 0.0]), persistent=False)

    def forward(self, views: torch.Tensor):
        B, V, H, W = views.shape
        x = views.reshape(B * V, 1, H, W)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.relu(self.conv2(x))
        x = x.reshape(B, V, *x.shape[1:]).amax(dim=1)  # pool over views

        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = self.feature_dropout(x)
        x = self.pooled_dropout(F.relu(self.conv4(x)))
        x = x.amax(dim=(2, 3))  # global max pool -> (B, 256)

        intensity = self.fc2(F.relu(self.fc1(x))) + self.intensity_bias
        latent = self.fc4(self.latent_dropout(self.fc3(x)))
        return latent, intensity


class Decoder(nn.Module):
    """Latent vector -> half-resolution image (B, 1, out_size, out_size), sine activations."""

    def __init__(self, out_size: int, latent_dim: int = 8):
        super().__init__()
        if out_size % 8:
            raise ValueError(f"Decoder output size must be divisible by 8, got {out_size}")
        self.seed_size = out_size // 8  # three stride-2 up-convolutions: 8x
        self.dense1 = nn.Linear(latent_dim, 64)
        self.dense2 = nn.Linear(64, self.seed_size ** 2 * 8)
        up = dict(kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv_t1 = nn.ConvTranspose2d(8, 256, **up)
        self.conv_t2 = nn.ConvTranspose2d(256, 128, **up)
        self.conv_t3 = nn.ConvTranspose2d(128, 128, **up)
        self.conv1 = nn.Conv2d(128, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=5, padding=2)
        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)
        self.drop3 = nn.Dropout2d(0.1)
        self.drop4 = nn.Dropout2d(0.3)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        s = self.seed_size
        x = self.dense2(self.dense1(latent))
        x = x.view(-1, s, s, 8).permute(0, 3, 1, 2)  # (B, 8, s, s)
        x = self.drop1(torch.sin(self.conv_t1(x)))
        x = self.drop2(torch.sin(self.conv_t2(x)))
        x = self.drop3(torch.sin(self.conv_t3(x)))
        x = self.drop4(torch.sin(self.conv1(x)))
        return OUTPUT_AMPLITUDE * torch.sin(self.conv2(x))


class BoostDecoder(nn.Module):
    """Learned 2x up-sampling of a half-resolution reconstruction (stage 3).

    out = w0 * detail + w1 * upsample_nearest(x) + w2, where ``detail`` is a
    learned full-resolution image and (w0, w1, w2) are predicted per particle
    (initialised around (1, 1, 0)).
    """

    def __init__(self):
        super().__init__()
        self.conv_t1 = nn.ConvTranspose2d(1, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=5, padding=2)
        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 3)
        self.register_buffer("weight_bias", torch.tensor([1.0, 1.0, 0.0]), persistent=False)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = self.drop1(torch.sin(self.conv_t1(x0)))
        x = self.drop2(torch.sin(self.conv1(x)))
        detail = OUTPUT_AMPLITUDE * torch.sin(self.conv2(x))
        base = F.interpolate(x0, scale_factor=2, mode="nearest")
        w = torch.sin(self.fc2(torch.sin(self.fc1(x.amax(dim=(2, 3)))))) + self.weight_bias
        w = w[:, :, None, None]
        return w[:, 0:1] * detail + w[:, 1:2] * base + w[:, 2:3]
