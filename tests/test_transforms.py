import math

import pytest
import torch

from cryo2dclass.filters import soft_circular_mask
from cryo2dclass.transforms import (
    ANGLE,
    MIRROR,
    align_to_reference,
    build_pose_grids,
    compose_poses,
    identity_poses,
    sampling_matrix,
    transform_images,
    undo_rotation,
)


def blob_image(size=64):
    """Smooth, asymmetric test image (no rotational or mirror symmetry)."""
    y, x = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32),
                          indexing="ij")
    c = (size - 1) / 2
    img = torch.zeros(size, size)
    for dx, dy, amp in ((0, 0, 1.0), (10, 0, 0.8), (18, 0, 0.6), (0, 9, 0.5)):
        img += amp * torch.exp(-((x - c - dx) ** 2 + (y - c - dy) ** 2) / 18)
    return img


def pose(angle_deg, sx, sy, mirror):
    return torch.tensor([[math.radians(angle_deg), sx, sy, mirror]])


def test_identity_pose_is_noop():
    img = torch.randn(2, 1, 32, 32)
    out = transform_images(img, identity_poses(2)[:, None])
    assert out.shape == (2, 1, 1, 32, 32)
    assert torch.allclose(out[:, 0], img, atol=1e-5)


def test_shift_moves_content_by_whole_pixels():
    img = torch.zeros(1, 1, 16, 16)
    img[0, 0, 8, 8] = 1.0
    out = transform_images(img, pose(0, 2, -3, 1)[:, None])[0, 0, 0]
    # out(p) = img(p + t): content moves by -t
    assert out[11, 6] == pytest.approx(1.0, abs=1e-5)


def test_output_shape_multiple_poses():
    img = torch.randn(3, 1, 48, 48)
    out = transform_images(img, torch.randn(3, 5, 4))
    assert out.shape == (3, 5, 1, 48, 48)


@pytest.mark.parametrize("mirror", [1.0, -1.0])
def test_align_to_reference_inverts_pose(mirror):
    ref = blob_image()
    p = pose(37, 3, -2, mirror)
    particle = transform_images(ref[None, None], p[:, None])[:, 0]
    back = align_to_reference(particle, p)[0, 0]
    inner = soft_circular_mask(64, 24, 0) > 0
    # two bilinear resamplings blur slightly; a wrong inverse would be off by ~O(1)
    assert (back - ref)[inner].abs().max() < 0.06
    wrong = align_to_reference(particle, pose(37, 3, -2, -mirror))[0, 0]
    assert (wrong - ref)[inner].abs().max() > 0.3


def test_undo_rotation_is_inverse_matrix():
    p = pose(71, 0, 0, -1)
    u = undo_rotation(p)
    a = sampling_matrix(p[:, ANGLE], p[:, MIRROR])
    a_inv = sampling_matrix(u[:, ANGLE], u[:, MIRROR])
    assert torch.allclose(a @ a_inv, torch.eye(2)[None], atol=1e-6)


@pytest.mark.parametrize("m", [1.0, -1.0])
@pytest.mark.parametrize("dm", [1.0, -1.0])
def test_compose_matches_matrix_algebra(m, dm):
    """A(-a'm', m') == A(-am, m) @ A(da, dm)^-1 for the composed pose."""
    cur = pose(50, 0, 0, m)
    delta = pose(-80, 0, 0, dm)
    new = compose_poses(cur, delta)
    lhs = sampling_matrix(-new[:, ANGLE] * new[:, MIRROR], new[:, MIRROR])
    rhs = sampling_matrix(-cur[:, ANGLE] * cur[:, MIRROR], cur[:, MIRROR]) @ \
        torch.linalg.inv(sampling_matrix(delta[:, ANGLE], delta[:, MIRROR]))
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_pose_search_recovers_true_pose():
    """Brute-force search + compose_poses recovers a particle's true pose, mirrored or not."""
    ref = blob_image()
    views, search = build_pose_grids(10.0, None, True, 1.0, 0)
    for true in (pose(120, 0, 0, -1), pose(250, 0, 0, 1)):
        particle = transform_images(ref[None, None], true[:, None])[:, 0]
        current = pose(30, 0, 0, 1)  # wrong starting pose
        target = transform_images(particle, undo_rotation(current)[:, None])[0, 0, 0]
        cands = transform_images(ref[None, None], search[None])[0, :, 0]
        best = (cands - target).pow(2).mean(dim=(-2, -1)).argmin()
        new = compose_poses(current, search[best][None])
        assert new[0, MIRROR] == true[0, MIRROR]
        err = (new[0, ANGLE] - true[0, ANGLE] + math.pi) % (2 * math.pi) - math.pi
        assert abs(math.degrees(err)) < 5.1


def test_grids():
    views, search = build_pose_grids(30.0, None, True, 2.0, 2)
    assert views.shape == (24, 4) and search.shape == (24 * 25, 4)
    assert torch.equal(views[0], torch.tensor([0.0, 0, 0, 1])) and torch.equal(search[0], views[0])
    views, search = build_pose_grids(6.0, 5, False, 4.0, 2)
    assert views.shape == (11, 4) and search.shape == (11 * 25, 4)
    assert math.degrees(views[:, ANGLE].abs().max()) == pytest.approx(30.0)
