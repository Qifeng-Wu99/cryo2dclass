"""Figures for a trained run (PNG files, no display needed).

=====================  =================================================================
``loss.png``           batch loss and per-round train / search loss
``poses.png``          distribution of final poses + how much poses moved per round
``latent_space.png``   PCA of the embeddings, coloured by loss and by cluster
``latent_map.png``     decoder output on a grid of K-means centres across the PCA plane
``reconstructions.png`` target / best-fitting reconstruction / residual for a few particles
``class_averages.png`` aligned class averages and decoded cluster centres (needs ``evaluate``)
=====================  =================================================================
"""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from . import checkpoint as ck  # noqa: E402
from .data import load_particles, read_stack  # noqa: E402
from .evaluate import decode_latents, default_device  # noqa: E402
from .transforms import ANGLE, MIRROR, SHIFT_X, SHIFT_Y  # noqa: E402

log = logging.getLogger("cryo2dclass")

FIGURES = ("loss", "poses", "latent", "map", "recon", "classes")


def _save(fig, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path)
    return path


def _image(ax, img, title=None):
    ax.imshow(img, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def _shade_warmup(ax, rounds):
    n_warm = sum(1 for r in rounds if r["warmup"])
    if n_warm:
        ax.axvspan(0.5, n_warm + 0.5, color="0.9", zorder=0, label="warm-up")


# ------------------------------------------------------------------------- figures

def plot_loss(run_dir: str, out_dir: str) -> Optional[str]:
    hist = ck.read_history(run_dir)
    rounds = hist["rounds"]
    if not rounds:
        log.warning("no history in %s", run_dir)
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(hist["batch_losses"], lw=0.5)
    axes[0].set(title="Batch loss", xlabel="step", ylabel="loss", yscale="log")
    x = [r["round"] + 1 for r in rounds]
    _shade_warmup(axes[1], rounds)
    axes[1].plot(x, [r["train_loss"] for r in rounds], "o-", label="train (mean over round)")
    axes[1].plot(x, [r["search_loss"] for r in rounds], "s--", label="pose search (all particles)")
    best = [r["round"] + 1 for r in rounds if r["best"]]
    axes[1].plot(best, [r["train_loss"] for r in rounds if r["best"]], "k*", ms=10, label="saved as best")
    axes[1].set(title="Loss per round", xlabel="round", ylabel="loss")
    axes[1].legend(fontsize=8)
    return _save(fig, out_dir, "loss.png")


def plot_poses(run_dir: str, out_dir: str, poses: torch.Tensor) -> str:
    p = poses.numpy()
    rounds = ck.read_history(run_dir)["rounds"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    axes[0, 0].hist(np.rad2deg(p[:, ANGLE]) % 360, bins=36, range=(0, 360))
    axes[0, 0].set(title="angle (deg)")
    axes[0, 1].hist(p[:, SHIFT_X], bins=21)
    axes[0, 1].set(title="shift x (px)")
    axes[0, 2].hist(p[:, SHIFT_Y], bins=21)
    axes[0, 2].set(title="shift y (px)")
    mirrored = int((p[:, MIRROR] < 0).sum())
    axes[0, 3].bar(["no mirror", "mirror"], [len(p) - mirrored, mirrored])
    axes[0, 3].set(title="mirror")
    x = [r["round"] + 1 for r in rounds]
    for ax, key, title in zip(axes[1], ("angle_change_deg", "shift_change_px", "mirror_flip_fraction"),
                              ("mean angle change (deg)", "mean shift change (px)", "fraction mirror flipped")):
        _shade_warmup(ax, rounds)
        ax.plot(x, [r[key] for r in rounds], "o-")
        ax.set(title=title, xlabel="round")
    axes[1, 3].axis("off")
    fig.suptitle("Particle poses (top: final; bottom: change per pose search)")
    return _save(fig, out_dir, "poses.png")


def plot_latent_space(out_dir: str, latent: np.ndarray, loss: np.ndarray,
                      labels: Optional[np.ndarray] = None) -> str:
    coords = PCA(n_components=2).fit_transform(latent)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    lo, hi = loss.mean() - 2 * loss.std(), loss.mean() + 2 * loss.std()
    sc = axes[0].scatter(coords[:, 0], -coords[:, 1], s=3, c=loss, cmap="RdBu_r", vmin=lo, vmax=hi, alpha=0.3)
    fig.colorbar(sc, ax=axes[0], label="particle loss")
    axes[0].set(title="Latent space (PCA) coloured by reconstruction error", xlabel="PC1", ylabel="-PC2")
    if labels is not None:
        for k in np.unique(labels):
            sel = labels == k
            axes[1].scatter(coords[sel, 0], -coords[sel, 1], s=3, alpha=0.3, label=f"cluster {k} ({sel.sum()})")
        axes[1].legend(markerscale=4, fontsize=8)
        axes[1].set(title="Clusters (from evaluate)", xlabel="PC1", ylabel="-PC2")
    else:
        axes[1].hexbin(coords[:, 0], -coords[:, 1], gridsize=60, cmap="viridis", mincnt=1)
        axes[1].set(title="Particle density (run `evaluate` to see clusters)", xlabel="PC1", ylabel="-PC2")
    return _save(fig, out_dir, "latent_space.png")


def plot_latent_map(out_dir: str, model, latent: np.ndarray, map_size: int = 4, seed: int = 0) -> str:
    """Decode a map_size x map_size grid of K-means centres laid out over the PCA plane."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(latent)
    s = map_size
    grid = np.indices((s, s)).reshape(2, -1).T[:, ::-1].astype(float)  # (col, row) per node
    grid = (grid - grid.mean(0)) / (grid.std(0) + 1e-8) * coords.std(0) * 1.5 + coords.mean(0)
    km = KMeans(n_clusters=s * s, init=grid, n_init=1, random_state=seed).fit(coords)
    images = decode_latents(model, pca.inverse_transform(km.cluster_centers_))
    sizes = np.bincount(km.labels_, minlength=s * s)
    fig, axes = plt.subplots(s, s, figsize=(2.2 * s, 2.3 * s))
    for i, ax in enumerate(np.atleast_1d(axes).flat):
        _image(ax, images[i], f"n={sizes[i]}")
    fig.suptitle("Latent map: decoded K-means centres (layout follows PC1 across, -PC2 down)")
    return _save(fig, out_dir, "latent_map.png")


@torch.no_grad()
def plot_reconstructions(out_dir: str, model, images: torch.Tensor, poses: torch.Tensor,
                         particle_ids: Sequence[int]) -> str:
    model.eval()
    device = next(model.parameters()).device
    ids = torch.as_tensor(list(particle_ids))
    out = model(images[ids].to(device), poses[ids].to(device))
    mse = out.residuals.pow(2).mean(dim=(-2, -1))
    best = mse.argmin(dim=1)
    resid = out.residuals[torch.arange(len(ids)), best].cpu().numpy()
    target = out.target.cpu().numpy()
    fig, axes = plt.subplots(len(ids), 3, figsize=(7.5, 2.5 * len(ids)), squeeze=False)
    for k, pid in enumerate(ids.tolist()):
        _image(axes[k, 0], target[k], f"particle {pid} (target)")
        _image(axes[k, 1], target[k] - resid[k], "best-fit reconstruction")
        _image(axes[k, 2], resid[k], f"residual, MSE={np.mean(resid[k] ** 2):.4f}")
    fig.suptitle("Target = masked, low-pass filtered particle in reference orientation")
    return _save(fig, out_dir, "reconstructions.png")


def plot_class_averages(out_dir: str, eval_dir: str, ncols: int = 8) -> Optional[str]:
    avg_path = os.path.join(eval_dir, "class_averages.mrcs")
    if not os.path.exists(avg_path):
        log.warning("no evaluation results in %s (run `cryo2dclass evaluate` first)", eval_dir)
        return None
    averages = read_stack(avg_path)
    decoded = read_stack(os.path.join(eval_dir, "decoded_centers.mrcs"))
    with open(os.path.join(eval_dir, "metrics.json")) as fh:
        sizes = json.load(fh)["cluster_sizes"]
    k = len(averages)
    ncols = min(k, ncols)
    blocks = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(2 * blocks, ncols, figsize=(2.2 * ncols, 4.6 * blocks), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i in range(k):
        b, c = divmod(i, ncols)
        _image(axes[2 * b, c], averages[i], f"class {i} (n={sizes[i]})")
        _image(axes[2 * b + 1, c], decoded[i], "decoded centre")
    fig.suptitle("Top: average of aligned particles | bottom: decoder output at the cluster centre")
    return _save(fig, out_dir, "class_averages.png")


# ------------------------------------------------------------------------- entry point

def visualize(run_dir: str, checkpoint: str = "best", figures: Iterable[str] = FIGURES,
              eval_dir: Optional[str] = None, particle_ids: Optional[Sequence[int]] = None,
              map_size: int = 4, out_dir: Optional[str] = None, data: Optional[str] = None,
              device=None) -> list:
    """Write the requested figures to ``out_dir`` (default ``<run>/figures``); returns their paths."""
    figures = list(figures)
    unknown = set(figures) - set(FIGURES)
    if unknown:
        raise ValueError(f"Unknown figure(s) {sorted(unknown)}; choose from {FIGURES}")
    out_dir = out_dir or os.path.join(run_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    eval_dir = eval_dir or os.path.join(run_dir, f"eval_{checkpoint}")
    device = torch.device(device) if device else default_device()
    cfg, model, poses, _ = ck.load_model(run_dir, checkpoint, device)

    # embeddings: prefer the evaluation of this checkpoint, else the training run's latest search
    labels = None
    if os.path.exists(os.path.join(eval_dir, "embeddings.npy")):
        latent = np.load(os.path.join(eval_dir, "embeddings.npy"))
        labels = np.load(os.path.join(eval_dir, "cluster_labels.npy"))
        loss = np.genfromtxt(os.path.join(eval_dir, "particles.csv"), delimiter=",", names=True)["loss"]
    else:
        latent = np.load(os.path.join(run_dir, ck.EMBEDDINGS_FILE))
        loss = np.load(os.path.join(run_dir, ck.LOSS_FILE))

    paths = []
    if "loss" in figures:
        paths.append(plot_loss(run_dir, out_dir))
    if "poses" in figures:
        paths.append(plot_poses(run_dir, out_dir, poses))
    if "latent" in figures:
        paths.append(plot_latent_space(out_dir, latent, loss, labels))
    if "map" in figures:
        paths.append(plot_latent_map(out_dir, model, latent, map_size))
    if "recon" in figures:
        images = torch.from_numpy(load_particles(data or cfg.data, cfg.normalize))
        ids = particle_ids if particle_ids is not None else np.linspace(0, len(images) - 1, 6).astype(int)
        paths.append(plot_reconstructions(out_dir, model, images, poses, ids))
    if "classes" in figures:
        paths.append(plot_class_averages(out_dir, eval_dir))
    return [p for p in paths if p]
