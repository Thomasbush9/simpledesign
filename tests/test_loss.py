import pytest
import torch

from simpledesign.models.loss import fape, joint_loss


def _helix(length=9):
    phase = torch.arange(length, dtype=torch.float32) * 1.1
    return torch.stack((0.23 * phase.cos(), 0.23 * phase.sin(), 0.14 * phase), dim=-1)


def test_fape_identical_and_independent_proper_rigid_motions_are_zero():
    coords = torch.stack((_helix(), _helix() * 1.3))
    mask = torch.ones(coords.shape[:2], dtype=torch.bool)
    rotate_pred = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
         [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]]
    )
    rotate_true = rotate_pred.flip(0)
    pred = coords @ rotate_pred + torch.tensor([[[2.0, -1.0, 3.0]], [[-2.0, 4.0, 1.0]]])
    true = coords @ rotate_true + torch.tensor([[[-1.0, 2.0, 1.0]], [[3.0, -2.0, 2.0]]])
    torch.testing.assert_close(fape(coords, coords, mask), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(fape(pred, true, mask), torch.tensor(0.0), atol=2e-6, rtol=0)


def test_fape_distinguishes_a_mirrored_nonplanar_fold_from_a_proper_rotation():
    true = _helix()[None]
    mask = torch.ones(true.shape[:2], dtype=torch.bool)
    proper = true * torch.tensor([-1.0, -1.0, 1.0])
    mirrored = true * torch.tensor([-1.0, 1.0, 1.0])
    torch.testing.assert_close(fape(proper, true, mask), torch.tensor(0.0), atol=1e-6, rtol=0)
    assert fape(mirrored, true, mask) > 0.01


def test_fape_frame_point_average_and_clamp_have_known_values():
    # Two frames, four points. Mirroring this right-angle chain changes only
    # one point per frame, by 0.4 nm: mean error = 2 * 0.4 / 8 = 0.1 nm.
    true = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0],
                          [0.2, 0.2, 0.0], [0.2, 0.2, 0.2]]])
    pred = true * torch.tensor([1.0, 1.0, -1.0])
    mask = torch.ones((1, 4), dtype=torch.bool)
    torch.testing.assert_close(fape(pred, true, mask), torch.tensor(0.1))
    torch.testing.assert_close(
        fape(pred, true, mask, clamp_distance=0.1, length_scale=0.5),
        torch.tensor(0.05),
    )


def test_fape_ragged_batch_equals_mean_of_unpadded_protein_losses():
    true = torch.stack((_helix(10), _helix(10) * 1.2))
    pred = true.clone()
    pred[0, 3] += torch.tensor([0.03, -0.04, 0.02])
    pred[1, 5] += torch.tensor([-0.04, 0.02, 0.05])
    mask = torch.arange(10)[None] < torch.tensor([[7], [10]])
    expected = (
        fape(pred[:1, :7], true[:1, :7], mask[:1, :7])
        + fape(pred[1:], true[1:], mask[1:])
    ) / 2
    # Padding may contain arbitrary values, including NaNs and infinities.
    pred[0, 7:] = float("nan")
    true[0, 7:] = float("inf")
    pred.requires_grad_()
    loss = fape(pred, true, mask)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    torch.testing.assert_close(pred.grad[0, 7:], torch.zeros((3, 3)))


def test_fape_does_not_create_frames_across_internal_masked_gaps():
    true = _helix(5)[None]
    pred = (true * torch.tensor([-1.0, 1.0, 1.0])).requires_grad_()
    mask = torch.tensor([[True, True, False, True, True]])
    loss = fape(pred, true, mask)
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    torch.testing.assert_close(pred.grad, torch.zeros_like(pred))


@pytest.mark.parametrize("length", [0, 1, 2])
def test_fape_short_proteins_return_differentiable_zero(length):
    true = _helix(length)[None]
    pred = true.clone().requires_grad_()
    loss = fape(pred, true, torch.ones((1, length), dtype=torch.bool))
    assert loss.shape == ()
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    torch.testing.assert_close(pred.grad, torch.zeros_like(pred))


@pytest.mark.parametrize("target_kind", ["collinear", "collapsed", "masked"])
def test_fape_no_valid_target_frames_returns_differentiable_zero(target_kind):
    true = torch.zeros((1, 7, 3))
    mask = torch.ones((1, 7), dtype=torch.bool)
    if target_kind == "collinear":
        true[0, :, 0] = torch.arange(7) * 0.38
    elif target_kind == "masked":
        true = _helix(7)[None]
        mask[:] = False
    pred = _helix(7)[None].requires_grad_()
    loss = fape(pred, true, mask)
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    torch.testing.assert_close(pred.grad, torch.zeros_like(pred))


@pytest.mark.parametrize("pred_kind", ["collapsed", "collinear", "near_collinear"])
def test_fape_degenerate_predictions_are_penalized_with_finite_gradients(pred_kind):
    true = _helix()[None]
    pred = torch.zeros_like(true)
    if pred_kind != "collapsed":
        pred[0, :, 0] = torch.arange(true.shape[1]) * 0.3
    if pred_kind == "near_collinear":
        pred[0, :, 1] = 1e-7 * torch.arange(true.shape[1]).square()
    pred.requires_grad_()
    loss = fape(pred, true, torch.ones(true.shape[:2], dtype=torch.bool))
    assert torch.isfinite(loss)
    assert 0 < loss <= 1
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_fape_perturbed_coordinates_have_useful_gradients_through_frames():
    true = _helix()[None]
    pred = true.clone()
    pred[0, 4] += torch.tensor([0.03, -0.02, 0.04])
    pred.requires_grad_()
    mask = torch.ones(true.shape[:2], dtype=torch.bool)
    loss = fape(pred, true, mask)
    assert 0 < loss < 1
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.norm() > 0
    # Finite differences include how moving this CA changes adjacent frames,
    # not just its role as a point; detached predicted frames give the wrong gradient.
    delta = torch.zeros_like(pred)
    delta[0, 4, 0] = 1e-4
    numerical = (
        fape(pred.detach() + delta, true, mask) - fape(pred.detach() - delta, true, mask)
    ) / (2e-4)
    torch.testing.assert_close(pred.grad[0, 4, 0], numerical, atol=2e-3, rtol=2e-2)
    # An actual descent step must reduce the geometric error.
    improved = fape(pred.detach() - 1e-4 * pred.grad, true, mask)
    assert improved < loss.detach()


def test_fape_computes_fp32_inside_autocast():
    true = _helix()[None].to(torch.bfloat16)
    pred = true.float()
    pred[0, 4] += torch.tensor([0.03, -0.02, 0.04])
    pred = pred.to(torch.bfloat16).requires_grad_()
    mask = torch.ones(true.shape[:2], dtype=torch.bool)
    expected = fape(pred.float(), true.float(), mask)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = fape(pred, true, mask)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(pred.grad).all()


def _joint_inputs():
    mask = torch.ones((1, 9), dtype=torch.bool)
    return {
        "logits": torch.zeros((1, 9, 33), requires_grad=True),
        "seq": torch.full((1, 9), 4, dtype=torch.long),
        "noise_mask": mask,
        "v_pred": torch.ones((1, 9, 3), requires_grad=True),
        "v_true": torch.zeros((1, 9, 3)),
        "struct_mask": mask,
        "t": torch.tensor([0.5]),
    }


def test_joint_loss_disabled_fape_preserves_original_objective_without_coordinates():
    inputs = _joint_inputs()
    total, parts = joint_loss(**inputs, lambda_seq=0.7, lambda_struct=1.3)
    # Uniform amino-acid logits give log(20) CE; velocity MSE is 3 per residue.
    expected = 0.7 * 0.5 * torch.log(torch.tensor(20.0)) + 1.3 * 3
    torch.testing.assert_close(total, expected)
    assert set(parts) == {"seq", "struct", "fape"}
    assert parts["fape"].shape == ()
    torch.testing.assert_close(parts["fape"], torch.tensor(0.0))
    assert not any(part.requires_grad for part in parts.values())
    # Disabled means the coordinate inputs are not inspected or evaluated.
    ignored, _ = joint_loss(**inputs, lambda_seq=0.7, lambda_struct=1.3,
                            beta_fape=0.0, x_pred=torch.tensor(float("nan")))
    torch.testing.assert_close(ignored, total)


def test_joint_loss_beta_scales_fape_and_backpropagates_to_clean_coordinates():
    inputs = _joint_inputs()
    true = _helix()[None]
    pred = true.clone()
    pred[0, 4] += torch.tensor([0.03, -0.02, 0.04])
    pred.requires_grad_()
    baseline, _ = joint_loss(**inputs, lambda_seq=0.7, lambda_struct=1.3)
    once, parts = joint_loss(**inputs, lambda_seq=0.7, lambda_struct=1.3,
                             beta_fape=1.0, x_pred=pred, x_true=true)
    twice, doubled_parts = joint_loss(**inputs, lambda_seq=0.7, lambda_struct=1.3,
                                      beta_fape=2.0, x_pred=pred, x_true=true)
    geometric = fape(pred, true, inputs["struct_mask"])
    assert geometric > 0
    torch.testing.assert_close(once, baseline + geometric)
    torch.testing.assert_close(twice, baseline + 2 * geometric)
    torch.testing.assert_close(parts["fape"], geometric.detach())
    torch.testing.assert_close(doubled_parts["fape"], parts["fape"])
    assert not any(part.requires_grad for part in parts.values())
    twice.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.norm() > 0


@pytest.mark.parametrize("missing", ["x_pred", "x_true", "both"])
def test_joint_loss_requires_both_coordinate_inputs_when_fape_is_enabled(missing):
    coords = {"x_pred": _helix()[None], "x_true": _helix()[None]}
    if missing == "both":
        coords.clear()
    else:
        del coords[missing]
    with pytest.raises(ValueError, match="x_pred and x_true"):
        joint_loss(**_joint_inputs(), beta_fape=1.0, **coords)


def test_joint_loss_rejects_negative_beta():
    with pytest.raises(ValueError, match="beta_fape"):
        joint_loss(**_joint_inputs(), beta_fape=-0.1)


def test_fape_excludes_only_degenerate_target_frames_not_the_whole_protein():
    # The first triple is collinear; only the frame centered at residue 2 counts.
    # Perturbing residue 0 leaves that frame fixed and changes one of four
    # frame-point distances by 0.1 nm, so the mean must be 0.025.
    true = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0],
                          [0.4, 0.0, 0.0], [0.4, 0.2, 0.0]]])
    pred = true.clone()
    pred[0, 0, 2] = 0.1
    mask = torch.ones((1, 4), dtype=torch.bool)
    torch.testing.assert_close(fape(pred, true, mask), torch.tensor(0.025))


def test_fape_proteins_without_frames_contribute_zero_to_batch_mean():
    true = torch.stack((_helix(), torch.zeros((9, 3))))
    pred = torch.stack((_helix() * torch.tensor([-1.0, 1.0, 1.0]), _helix()))
    mask = torch.ones((2, 9), dtype=torch.bool)
    expected = fape(pred[:1], true[:1], mask[:1]) / 2
    torch.testing.assert_close(fape(pred, true, mask), expected)
