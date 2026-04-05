"""Tests for YOLOatr loss functions."""

import torch

from anima_yoloatr.losses import ComputeLoss, FocalBCELoss, bbox_iou
from anima_yoloatr.model import build_model


class TestBboxIoU:
    """Test IoU computation."""

    def test_perfect_overlap(self):
        box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        iou = bbox_iou(box, box, xywh=True, ciou=False)
        assert iou.item() > 0.999

    def test_no_overlap(self):
        box1 = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
        box2 = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
        iou = bbox_iou(box1, box2, xywh=True, ciou=False)
        assert iou.item() < 0.01

    def test_ciou_range(self):
        box1 = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
        box2 = torch.tensor([[0.6, 0.6, 0.3, 0.3]])
        ciou = bbox_iou(box1, box2, xywh=True, ciou=True)
        assert -1.0 < ciou.item() < 1.0

    def test_xyxy_format(self):
        box1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        box2 = torch.tensor([[0.5, 0.5, 1.5, 1.5]])
        iou = bbox_iou(box1, box2, xywh=False, ciou=False)
        assert 0.1 < iou.item() < 0.5


class TestFocalBCELoss:
    """Test focal BCE loss."""

    def test_output_positive(self):
        loss_fn = FocalBCELoss(gamma=0.3, alpha=0.25)
        pred = torch.randn(10, 4)
        target = torch.zeros(10, 4)
        target[:5] = 1.0
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_gamma_effect(self):
        pred = torch.randn(100, 4)
        target = torch.zeros(100, 4)
        target[:50] = 1.0
        loss_low = FocalBCELoss(gamma=0.0)(pred, target)
        loss_high = FocalBCELoss(gamma=2.0)(pred, target)
        # Higher gamma focuses more on hard examples, typically lower loss
        assert loss_low.item() != loss_high.item()


class TestComputeLoss:
    """Test full loss computation."""

    def test_with_targets(self):
        model = build_model(num_classes=4)
        model.eval()
        compute_loss = ComputeLoss(model)

        x = torch.randn(2, 3, 640, 640)
        preds = model(x, decode=False)

        # Create targets: [batch_idx, class, cx, cy, w, h]
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.1, 0.1],
            [0, 1, 0.3, 0.3, 0.2, 0.15],
            [1, 2, 0.7, 0.7, 0.12, 0.08],
        ])

        loss, loss_dict = compute_loss(preds, targets)
        assert loss.item() > 0
        assert "box" in loss_dict
        assert "obj" in loss_dict
        assert "cls" in loss_dict
        assert "total" in loss_dict

    def test_no_targets(self):
        model = build_model(num_classes=4)
        model.eval()
        compute_loss = ComputeLoss(model)

        x = torch.randn(1, 3, 640, 640)
        preds = model(x, decode=False)
        targets = torch.zeros((0, 6))

        loss, loss_dict = compute_loss(preds, targets)
        # With no targets, box and cls should be 0, obj should still exist
        assert loss.item() >= 0
        assert loss_dict["box"] == 0.0

    def test_gradient_flows(self):
        model = build_model(num_classes=4)
        model.train()
        compute_loss = ComputeLoss(model)

        x = torch.randn(1, 3, 640, 640)
        preds = model(x, decode=False)
        targets = torch.tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]])

        loss, _ = compute_loss(preds, targets)
        loss.backward()

        # Check gradients exist
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad
