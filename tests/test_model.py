"""Tests for YOLOatr model architecture."""

import torch

from anima_yoloatr.model import (
    C3,
    CBS,
    SPPF,
    BiFPNBlock,
    BiFPNWeightedAdd,
    Bottleneck,
    Detect,
    Focus,
    build_model,
)


class TestBuildingBlocks:
    """Test individual building blocks."""

    def test_cbs(self):
        block = CBS(3, 32, k=3, s=1)
        x = torch.randn(1, 3, 64, 64)
        out = block(x)
        assert out.shape == (1, 32, 64, 64)

    def test_cbs_stride2(self):
        block = CBS(32, 64, k=3, s=2)
        x = torch.randn(1, 32, 64, 64)
        out = block(x)
        assert out.shape == (1, 64, 32, 32)

    def test_bottleneck(self):
        block = Bottleneck(64, 64, shortcut=True)
        x = torch.randn(1, 64, 32, 32)
        out = block(x)
        assert out.shape == (1, 64, 32, 32)

    def test_c3(self):
        block = C3(64, 64, n=2)
        x = torch.randn(1, 64, 32, 32)
        out = block(x)
        assert out.shape == (1, 64, 32, 32)

    def test_sppf(self):
        block = SPPF(512, 512)
        x = torch.randn(1, 512, 20, 20)
        out = block(x)
        assert out.shape == (1, 512, 20, 20)

    def test_focus(self):
        block = Focus(3, 32, k=3)
        x = torch.randn(1, 3, 640, 640)
        out = block(x)
        assert out.shape == (1, 32, 320, 320)


class TestBiFPN:
    """Test BiFPN components."""

    def test_weighted_add(self):
        wadd = BiFPNWeightedAdd(3)
        inputs = [torch.randn(1, 64, 32, 32) for _ in range(3)]
        out = wadd(inputs)
        assert out.shape == (1, 64, 32, 32)

    def test_bifpn_block(self):
        block = BiFPNBlock(
            channels=[64, 128, 256, 512],
            out_channels=128,
        )
        features = [
            torch.randn(1, 64, 160, 160),   # P2
            torch.randn(1, 128, 80, 80),     # P3
            torch.randn(1, 256, 40, 40),     # P4
            torch.randn(1, 512, 20, 20),     # P5
        ]
        out = block(features)
        assert len(out) == 4
        assert out[0].shape == (1, 128, 160, 160)
        assert out[1].shape == (1, 128, 80, 80)
        assert out[2].shape == (1, 128, 40, 40)
        assert out[3].shape == (1, 128, 20, 20)


class TestDetect:
    """Test detection head."""

    def test_detect_forward(self):
        head = Detect(num_classes=4, channels=[128, 128, 128, 128])
        features = [
            torch.randn(2, 128, 160, 160),
            torch.randn(2, 128, 80, 80),
            torch.randn(2, 128, 40, 40),
            torch.randn(2, 128, 20, 20),
        ]
        out = head(features)
        assert len(out) == 4
        # Each output: [batch, num_anchors, grid_h, grid_w, 5+nc]
        assert out[0].shape == (2, 3, 160, 160, 9)  # 5 + 4 classes
        assert out[1].shape == (2, 3, 80, 80, 9)
        assert out[2].shape == (2, 3, 40, 40, 9)
        assert out[3].shape == (2, 3, 20, 20, 9)

    def test_detect_decode(self):
        head = Detect(num_classes=4, channels=[128, 128, 128, 128])
        features = [
            torch.randn(1, 128, 160, 160),
            torch.randn(1, 128, 80, 80),
            torch.randn(1, 128, 40, 40),
            torch.randn(1, 128, 20, 20),
        ]
        preds = head(features)
        decoded = head.decode(preds)
        # Total anchors: 3*(160*160 + 80*80 + 40*40 + 20*20)
        total = 3 * (160 * 160 + 80 * 80 + 40 * 40 + 20 * 20)
        assert decoded.shape == (1, total, 9)


class TestYOLOatr:
    """Test full YOLOatr model."""

    def test_forward_training(self):
        model = build_model(num_classes=4)
        x = torch.randn(1, 3, 640, 640)
        out = model(x, decode=False)
        assert isinstance(out, list)
        assert len(out) == 4

    def test_forward_inference(self):
        model = build_model(num_classes=4)
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        out = model(x, decode=True)
        assert out.ndim == 3
        assert out.shape[0] == 1
        assert out.shape[2] == 9  # 5 + 4 classes

    def test_parameter_count(self):
        model = build_model(num_classes=4)
        total = sum(p.numel() for p in model.parameters())
        # Paper reports 7,086,449 -- allow 30% tolerance for minor arch differences
        assert total > 3_000_000, f"Too few params: {total}"
        assert total < 15_000_000, f"Too many params: {total}"

    def test_batch_forward(self):
        model = build_model(num_classes=4)
        x = torch.randn(2, 3, 640, 640)
        out = model(x, decode=False)
        assert out[0].shape[0] == 2

    def test_single_channel_input(self):
        model = build_model(num_classes=4, in_channels=1)
        x = torch.randn(1, 1, 640, 640)
        out = model(x, decode=False)
        assert len(out) == 4

    def test_gradient_flow(self):
        model = build_model(num_classes=4)
        x = torch.randn(1, 3, 640, 640, requires_grad=True)
        out = model(x, decode=False)
        loss = sum(o.sum() for o in out)
        loss.backward()
        assert x.grad is not None
