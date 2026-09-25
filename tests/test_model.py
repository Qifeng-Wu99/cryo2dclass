import pytest
import torch

from cryo2dclass.config import resolve_config
from cryo2dclass.model import AlignedAutoencoder
from cryo2dclass.transforms import identity_poses

BOX = 32


@pytest.fixture(params=[1, 2, 3])
def model(request):
    torch.manual_seed(0)
    return AlignedAutoencoder(BOX, resolve_config(request.param, box_size=BOX))


def test_forward_shapes_and_backward(model):
    x, p = torch.randn(3, BOX, BOX), identity_poses(3)
    out = model(x, p)
    h = BOX // model.scale
    assert out.target.shape == (3, h, h)
    assert out.latent.shape == (3, 8)
    assert out.recon.shape == (3, 1, h, h)
    assert out.residuals.shape == (3, len(model.search_poses), h, h)
    loss = model.loss(out)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters())


def test_search_updates_poses_in_input_pixels(model):
    res = model.search(torch.randn(4, BOX, BOX), identity_poses(4))
    assert res.poses.shape == (4, 4) and res.loss.shape == (4,) and res.latent.shape == (4, 8)
    assert model.training  # search restores the training flag
    limited = model.limit_shifts(res.poses, max_shift=BOX / 4)
    assert torch.all(limited[:, 1:3] % model.scale == 0)  # whole working pixels


def test_target_is_rms_normalised(model):
    out = model(torch.randn(2, BOX, BOX) * 7, identity_poses(2))
    rms = out.target.pow(2).mean(dim=(-2, -1)).sqrt()
    assert torch.allclose(rms, torch.ones(2), atol=1e-4)


def test_state_dict_has_only_weights(model):
    keys = model.state_dict().keys()
    assert not any(k.endswith(("mask", "poses", "kernel", "bias_")) for k in keys)
    assert any(k.startswith("boost.") for k in keys) == (model.boost is not None)


def test_stage2_weights_load_into_stage3():
    m2 = AlignedAutoencoder(BOX, resolve_config(2, box_size=BOX))
    m3 = AlignedAutoencoder(BOX, resolve_config(3, box_size=BOX))
    with pytest.raises(RuntimeError):
        m3.load_weights(m2.state_dict())
    m3.load_weights(m2.state_dict(), allow_missing_boost=True)
    assert torch.equal(m3.encoder.conv1.weight, m2.encoder.conv1.weight)


def test_rejects_bad_box():
    with pytest.raises(ValueError):
        AlignedAutoencoder(40, resolve_config(1))
