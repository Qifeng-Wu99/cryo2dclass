"""Embed particles with a trained model, cluster them and summarise the classes.

Outputs (default directory ``<run>/eval_<checkpoint>/``)::

    particles.csv          one row per particle: cluster, loss, PCA coords, pose (+ true label)
    embeddings.npy         (N, latent_dim) latent vectors
    cluster_labels.npy     (N,) cluster index per particle
    class_averages.mrcs    (K, box, box) mean of the aligned particles of each cluster
    decoded_centers.mrcs   (K, h, h) decoder output at each cluster centre
    metrics.json           cluster sizes (+ accuracy / ARI / NMI if labels were given)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm.auto import tqdm

from . import checkpoint as ck
from .data import load_particles, save_stack
from .model import AlignedAutoencoder, SearchResult
from .transforms import align_to_reference

log = logging.getLogger("cryo2dclass")


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embed(model: AlignedAutoencoder, images: torch.Tensor, poses: torch.Tensor,
          batch_size: int = 64, device=None) -> SearchResult:
    """Latent vectors, reconstruction errors and refined poses for all particles (on CPU)."""
    device = device or next(model.parameters()).device
    parts = []
    for start in tqdm(range(0, len(images), batch_size), desc="embedding", leave=False, disable=None):
        sl = slice(start, start + batch_size)
        parts.append(model.search(images[sl].to(device), poses[sl].to(device)))
    return SearchResult(*(torch.cat([getattr(p, f) for p in parts]).cpu() for f in SearchResult._fields))


def cluster(latent: np.ndarray, n_clusters: int, pca_dims: int = 2, seed: int = 0) -> Dict[str, np.ndarray]:
    """K-means on the latent vectors (optionally after PCA to ``pca_dims`` dimensions).

    Returns labels (N,), coords (N, pca_dims or D) used for clustering, and the
    cluster centres mapped back to latent space (K, D).
    """
    pca = PCA(n_components=pca_dims) if pca_dims else None
    coords = pca.fit_transform(latent) if pca else latent
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(coords)
    centers = pca.inverse_transform(km.cluster_centers_) if pca else km.cluster_centers_
    return {
        "labels": km.labels_.astype(np.int64),
        "coords": coords,
        "centers_latent": centers.astype(np.float32),
        "explained_variance": (pca.explained_variance_ratio_ if pca else np.ones(latent.shape[1])),
    }


def clustering_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    """Accuracy after optimal (Hungarian) matching of clusters to classes, ARI and NMI."""
    classes, t = np.unique(true, return_inverse=True)
    clusters, p = np.unique(pred, return_inverse=True)
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    np.add.at(contingency, (t, p), 1)
    rows, cols = linear_sum_assignment(-contingency)
    return {
        "accuracy": float(contingency[rows, cols].sum() / len(true)),
        "adjusted_rand_index": float(adjusted_rand_score(true, pred)),
        "normalized_mutual_info": float(normalized_mutual_info_score(true, pred)),
        "cluster_to_class": {int(clusters[c]): int(classes[r]) for r, c in zip(rows, cols)},
        "contingency": {"classes": classes.tolist(), "clusters": clusters.tolist(),
                        "counts": contingency.tolist()},
    }


@torch.no_grad()
def class_averages(images: torch.Tensor, poses: torch.Tensor, labels: np.ndarray, n_clusters: int,
                   batch_size: int = 256, device="cpu") -> np.ndarray:
    """(K, H, W) average of the particles of each cluster after aligning them to the reference frame."""
    sums = torch.zeros(n_clusters, *images.shape[1:], device=device)
    lab = torch.from_numpy(labels).to(device)
    for start in range(0, len(images), batch_size):
        sl = slice(start, start + batch_size)
        aligned = align_to_reference(images[sl, None].to(device), poses[sl].to(device))[:, 0]
        sums.index_add_(0, lab[sl], aligned)
    counts = torch.bincount(lab, minlength=n_clusters).clamp_min(1).view(-1, 1, 1)
    return (sums / counts).cpu().numpy()


@torch.no_grad()
def decode_latents(model: AlignedAutoencoder, latent: np.ndarray) -> np.ndarray:
    """Decoder images (K, h, w) for latent vectors (K, D)."""
    model.eval()
    device = next(model.parameters()).device
    return model.decode(torch.as_tensor(latent, dtype=torch.float32, device=device))[:, 0].cpu().numpy()


def evaluate(run_dir: str, checkpoint: str = "best", data: Optional[str] = None, n_clusters: int = 2,
             labels: Optional[np.ndarray] = None, pca_dims: int = 2, out_dir: Optional[str] = None,
             batch_size: Optional[int] = None, device=None, seed: int = 0) -> Dict[str, object]:
    """Evaluate a trained run; see the module docstring for the files written.

    Args:
        run_dir: run directory produced by ``train``.
        checkpoint: best, last or round_NNN.
        data: particle stack (default: the one recorded in the run's config).
        n_clusters: number of K-means clusters.
        labels: optional ground-truth class per particle, for accuracy / ARI / NMI.
        pca_dims: cluster in the first ``pca_dims`` principal components (0 = full latent space).
    Returns:
        the metrics dictionary (also written to metrics.json).
    """
    device = torch.device(device) if device else default_device()
    cfg, model, poses, ckpt = ck.load_model(run_dir, checkpoint, device)
    out_dir = out_dir or os.path.join(run_dir, f"eval_{checkpoint}")
    os.makedirs(out_dir, exist_ok=True)

    images = torch.from_numpy(load_particles(data or cfg.data, cfg.normalize))
    if len(images) != len(poses):
        raise ValueError(f"The stack has {len(images)} particles but the checkpoint has poses for {len(poses)}")
    if labels is not None and len(labels) != len(images):
        raise ValueError(f"{len(labels)} labels given for {len(images)} particles")

    log.info("Evaluating %s (%s checkpoint, round %d): %d particles, k=%d",
             run_dir, checkpoint, ckpt["round"], len(images), n_clusters)
    result = embed(model, images, poses, batch_size or cfg.batch_size, device)
    latent, loss = result.latent.numpy(), result.loss.numpy()
    clus = cluster(latent, n_clusters, pca_dims, seed)
    lab = clus["labels"]

    averages = class_averages(images, poses, lab, n_clusters, device=device)
    decoded = decode_latents(model, clus["centers_latent"])

    np.save(os.path.join(out_dir, "embeddings.npy"), latent)
    np.save(os.path.join(out_dir, "cluster_labels.npy"), lab)
    save_stack(os.path.join(out_dir, "class_averages.mrcs"), averages)
    save_stack(os.path.join(out_dir, "decoded_centers.mrcs"), decoded)
    extra = {"cluster": lab, "loss": loss}
    for d in range(clus["coords"].shape[1] if pca_dims else 0):
        extra[f"pc{d + 1}"] = clus["coords"][:, d]
    if labels is not None:
        extra["true_label"] = np.asarray(labels)
    ck.save_poses_csv(os.path.join(out_dir, "particles.csv"), poses, extra)

    metrics: Dict[str, object] = {
        "run": os.path.abspath(run_dir), "checkpoint": checkpoint, "stage": ckpt["stage"],
        "round": ckpt["round"], "n_particles": int(len(images)), "n_clusters": int(n_clusters),
        "pca_dims": int(pca_dims), "cluster_sizes": np.bincount(lab, minlength=n_clusters).tolist(),
        "mean_particle_loss": float(loss.mean()),
        "pca_explained_variance": [float(v) for v in clus["explained_variance"]],
    }
    if labels is not None:
        metrics.update(clustering_metrics(np.asarray(labels), lab))
        log.info("accuracy %.4f | ARI %.4f | NMI %.4f", metrics["accuracy"],
                 metrics["adjusted_rand_index"], metrics["normalized_mutual_info"])
    log.info("cluster sizes: %s", metrics["cluster_sizes"])
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Evaluation written to %s", out_dir)
    return metrics
