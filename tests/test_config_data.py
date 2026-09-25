import argparse

import numpy as np
import pytest

from cryo2dclass.config import (
    STAGE_PRESETS,
    TrainConfig,
    add_config_arguments,
    load_yaml,
    namespace_overrides,
    resolve_config,
    save_yaml,
)
from cryo2dclass.data import labels_from_counts, load_labels, load_particles, save_stack, stack_shape

# ----------------------------------------------------------------------------- config

def test_presets_resolve_for_reference_box():
    for stage, preset in STAGE_PRESETS.items():
        cfg = resolve_config(stage, box_size=128)
        for key, value in preset.items():
            assert getattr(cfg, key) == value


def test_pixel_values_scale_with_box():
    cfg = resolve_config(2, box_size=64)
    assert cfg.mask_radius == 30 and cfg.shift_step == 2 and cfg.angle_step == 10


def test_precedence_overrides_win():
    cfg = resolve_config(1, {"lr": 1e-3, "mask_radius": 20.0}, box_size=64)
    assert cfg.lr == 1e-3 and cfg.mask_radius == 20.0


def test_unknown_key_suggests_fix():
    with pytest.raises(KeyError, match="learning|lr"):
        resolve_config(1, {"lrr": 1e-3})


def test_invalid_combination():
    with pytest.raises(ValueError):
        resolve_config(1, {"boost_decoder": True})


def test_yaml_roundtrip(tmp_path):
    cfg = resolve_config(3, {"data": "x.mrcs", "out": str(tmp_path)})
    save_yaml(cfg, tmp_path / "c.yaml")
    assert TrainConfig(**load_yaml(tmp_path / "c.yaml")) == cfg


def test_cli_flags_only_contain_explicit_values():
    p = argparse.ArgumentParser()
    add_config_arguments(p)
    ns = p.parse_args(["--lr", "0.1", "--no-search-mirror", "--local-angle-steps", "none"])
    assert namespace_overrides(ns) == {"lr": 0.1, "search_mirror": False, "local_angle_steps": None}


# ----------------------------------------------------------------------------- data

@pytest.mark.parametrize("ext", [".npy", ".mrcs"])
def test_stack_roundtrip(tmp_path, ext):
    arr = np.random.rand(5, 32, 32).astype(np.float32)
    path = str(tmp_path / f"p{ext}")
    save_stack(path, arr)
    assert stack_shape(path) == (5, 32, 32)
    np.testing.assert_allclose(load_particles(path), arr)


def test_normalize(tmp_path):
    save_stack(str(tmp_path / "p.npy"), np.random.rand(3, 16, 16) * 5 + 2)
    x = load_particles(str(tmp_path / "p.npy"), normalize=True)
    np.testing.assert_allclose(x.mean(axis=(1, 2)), 0, atol=1e-5)
    np.testing.assert_allclose(x.std(axis=(1, 2)), 1, atol=1e-4)


@pytest.mark.parametrize("shape,msg", [((4, 30, 30), "divisible by 16"), ((4, 32, 48), "square"),
                                       ((32, 32), "stack")])
def test_bad_stacks_are_rejected(tmp_path, shape, msg):
    np.save(tmp_path / "bad.npy", np.zeros(shape, np.float32))
    with pytest.raises(ValueError, match=msg):
        load_particles(str(tmp_path / "bad.npy"))


def test_labels(tmp_path):
    (tmp_path / "l.txt").write_text("0\n1\n1\n")
    (tmp_path / "l.csv").write_text("particle,label\n0,2\n1,0\n")
    np.testing.assert_array_equal(load_labels(str(tmp_path / "l.txt")), [0, 1, 1])
    np.testing.assert_array_equal(load_labels(str(tmp_path / "l.csv")), [2, 0])
    np.testing.assert_array_equal(labels_from_counts([2, 3]), [0, 0, 1, 1, 1])
