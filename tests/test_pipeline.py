"""End-to-end smoke test of the command-line workflow on a tiny synthetic stack (CPU, ~1 min)."""

import json
import os

import numpy as np
import pytest

from cryo2dclass.cli import main
from cryo2dclass.toy import make_toy_dataset

FAST = ["--n-rounds", "2", "--epochs-per-round", "1", "--warmup-rounds", "1", "--batch-size", "16"]


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy")
    return make_toy_dataset(str(root), n=40, box=32, n_classes=2), root


def test_stagewise_cli(toy):
    data, root = toy
    s1, s2 = str(root / "s1"), str(root / "s2")
    main(["train", "--stage", "1", "--data", data, "--out", s1, *FAST])
    main(["train", "--stage", "2", "--data", data, "--out", s2, "--init-from", s1,
          "--train-on-selection", *FAST])
    for name in ("config.yaml", "history.json", "poses.csv", "embeddings.npy", "particle_loss.npy",
                 "selected_particles.npy", "checkpoints/best.pt", "checkpoints/last.pt",
                 "checkpoints/round_001.pt"):
        assert os.path.exists(os.path.join(s2, name)), name
    assert len(np.load(os.path.join(s1, "selected_particles.npy"))) == 36  # keep_fraction 0.9

    # resume: extend by one round
    main(["train", "--resume", "--out", s2, "--n-rounds", "3"])
    with open(os.path.join(s2, "history.json")) as fh:
        assert len(json.load(fh)["rounds"]) == 3

    # refuse to overwrite an existing run
    with pytest.raises(SystemExit):
        main(["train", "--stage", "2", "--data", data, "--out", s2, *FAST])

    main(["evaluate", s2, "--n-clusters", "2", "--labels", str(root / "labels.txt"), "--device", "cpu"])
    ev = os.path.join(s2, "eval_best")
    with open(os.path.join(ev, "metrics.json")) as fh:
        metrics = json.load(fh)
    assert 0.5 <= metrics["accuracy"] <= 1 and sum(metrics["cluster_sizes"]) == 40
    for name in ("particles.csv", "class_averages.mrcs", "decoded_centers.mrcs", "cluster_labels.npy"):
        assert os.path.exists(os.path.join(ev, name)), name

    main(["visualize", s2, "--map-size", "2", "--device", "cpu"])
    for fig in ("loss", "poses", "latent_space", "latent_map", "reconstructions", "class_averages"):
        assert os.path.exists(os.path.join(s2, "figures", f"{fig}.png")), fig


def test_pipeline_command(toy):
    data, root = toy
    out = str(root / "pipe")
    main(["pipeline", "--data", data, "--out", out, "--stages", "1", "3", "--n-clusters", "2",
          "--label-counts", "20", "20", *FAST])
    assert os.path.exists(os.path.join(out, "stage3", "eval_best", "metrics.json"))
    assert os.path.exists(os.path.join(out, "stage3", "figures", "class_averages.png"))
    # rerunning with --resume is a no-op for finished stages
    main(["pipeline", "--data", data, "--out", out, "--stages", "1", "3", "--resume", *FAST])
