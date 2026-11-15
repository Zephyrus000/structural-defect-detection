import pytest
import torch

from structural_defect_detection.config import ConfigLoader
from structural_defect_detection.losses import CompositeLoss
from structural_defect_detection.losses.light_topology import euler_curve, soft_cldice_loss


def test_euler_curve_counts_components_and_holes_at_multiple_thresholds():
    ring = torch.zeros(4, 4)
    ring[:3, :3] = 0.9
    ring[1, 1] = 0.4
    separate = torch.zeros(4, 4)
    separate[0, 0], separate[3, 3] = 0.9, 0.9
    diagonal = torch.eye(4)
    images = torch.stack((ring, separate, diagonal))
    curve = euler_curve(images, torch.tensor([0.25, 0.5]), smooth=False)
    torch.testing.assert_close(curve, torch.tensor([[1., 0.], [2., 2.], [1., 1.]]))


def test_cldice_penalizes_missing_centerline():
    target = torch.zeros((1, 1, 16, 16))
    target[..., 8, 2:14] = 1
    broken = target.clone()
    broken[..., 8, 7:9] = 0
    valid = torch.ones_like(target, dtype=torch.bool)
    assert soft_cldice_loss(target, target, valid).item() == pytest.approx(0)
    assert soft_cldice_loss(broken, target, valid).item() == pytest.approx(1 / 11)


@pytest.mark.parametrize("name", ["cldice", "ecc"])
def test_topology_term_backpropagates_and_respects_ignore(name):
    config = ConfigLoader.from_dict({"loss": {"ph": {"weight": 0}, f"{name}_weight": 0.1}})
    criterion = CompositeLoss(config.loss)
    target = torch.zeros((2, 1, 32, 64))
    target[..., 16, 4:27] = 1
    logits = (target * 3 - 1.5).requires_grad_()
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[0, ..., 32:] = False
    valid[1] = False
    result = criterion(logits, target, valid)
    grad = torch.autograd.grad(result.components[name], logits)[0]
    assert torch.isfinite(grad).all()
    assert grad[valid].abs().sum() > 0
    assert grad[~valid].count_nonzero() == 0
    assert criterion(logits, target, torch.zeros_like(valid)).total.item() == 0
