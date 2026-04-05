"""YOLOatr model architecture.

Modified YOLOv5s with:
1. BiFPN neck (replaces PANet)
2. Extra P2 small-object detection head (stride=4)
3. 4 detection scales: P2(160x160), P3(80x80), P4(40x40), P5(20x20)

Paper: arxiv 2507.11267
Parameters: ~7.1M | GFLOPs: ~16.4
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def autopad(k: int, p: int | None = None) -> int:
    """Auto-pad to maintain spatial size for 'same' convolution."""
    if p is None:
        p = k // 2
    return p


class CBS(nn.Module):
    """Conv-BatchNorm-SiLU block (standard YOLOv5 building block)."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck: 1x1 conv -> 3x3 conv with optional residual."""

    def __init__(self, c_in: int, c_out: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = CBS(c_in, c_hidden, 1, 1)
        self.cv2 = CBS(c_hidden, c_out, 3, 1)
        self.add = shortcut and c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions (YOLOv5 C3 block)."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 0.5,
    ):
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = CBS(c_in, c_hidden, 1, 1)
        self.cv2 = CBS(c_in, c_hidden, 1, 1)
        self.cv3 = CBS(2 * c_hidden, c_out, 1, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_hidden, c_hidden, shortcut, e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (YOLOv5 SPPF)."""

    def __init__(self, c_in: int, c_out: int, k: int = 5):
        super().__init__()
        c_hidden = c_in // 2
        self.cv1 = CBS(c_in, c_hidden, 1, 1)
        self.cv2 = CBS(c_hidden * 4, c_out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


class Focus(nn.Module):
    """Focus layer: space-to-depth then conv (YOLOv5 stem).

    Slices input into 4 sub-images and concatenates along channel dim.
    """

    def __init__(self, c_in: int, c_out: int, k: int = 1, s: int = 1):
        super().__init__()
        self.conv = CBS(c_in * 4, c_out, k, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Space-to-depth: [B, C, H, W] -> [B, 4C, H/2, W/2]
        return self.conv(
            torch.cat(
                (
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ),
                dim=1,
            )
        )


# ---------------------------------------------------------------------------
# BiFPN -- Weighted Bi-directional Feature Pyramid Network
# ---------------------------------------------------------------------------


class BiFPNWeightedAdd(nn.Module):
    """Fast normalized weighted addition for BiFPN.

    w_i' = w_i / (sum(w_j) + eps) -- fast fusion from EfficientDet.
    """

    def __init__(self, num_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        w = functional.relu(self.w)
        w_sum = w.sum() + self.eps
        weighted = sum(w_i * x_i for w_i, x_i in zip(w, inputs, strict=False))
        return weighted / w_sum


class BiFPNBlock(nn.Module):
    """Single BiFPN block: top-down + bottom-up with weighted fusion.

    Operates on 4 feature levels: P2, P3, P4, P5.
    """

    def __init__(self, channels: list[int], out_channels: int):
        super().__init__()
        # Lateral convolutions to unify channel dimensions
        self.lateral_convs = nn.ModuleList(
            [CBS(c, out_channels, 1, 1) for c in channels]
        )

        # Top-down pathway (P5 -> P4 -> P3 -> P2)
        self.td_fuse_p4 = BiFPNWeightedAdd(2)
        self.td_fuse_p3 = BiFPNWeightedAdd(2)
        self.td_fuse_p2 = BiFPNWeightedAdd(2)
        self.td_conv_p4 = CBS(out_channels, out_channels, 3, 1)
        self.td_conv_p3 = CBS(out_channels, out_channels, 3, 1)
        self.td_conv_p2 = CBS(out_channels, out_channels, 3, 1)

        # Bottom-up pathway (P2 -> P3 -> P4 -> P5)
        self.bu_fuse_p3 = BiFPNWeightedAdd(3)  # original + td + bu
        self.bu_fuse_p4 = BiFPNWeightedAdd(3)
        self.bu_fuse_p5 = BiFPNWeightedAdd(2)
        self.bu_conv_p3 = CBS(out_channels, out_channels, 3, 1)
        self.bu_conv_p4 = CBS(out_channels, out_channels, 3, 1)
        self.bu_conv_p5 = CBS(out_channels, out_channels, 3, 1)

    def forward(
        self, features: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            features: [P2, P3, P4, P5] feature maps

        Returns:
            [P2_out, P3_out, P4_out, P5_out] fused features
        """
        # Apply lateral convolutions
        p2, p3, p4, p5 = [conv(f) for conv, f in zip(self.lateral_convs, features, strict=False)]

        # Top-down pathway
        p4_td = self.td_conv_p4(
            self.td_fuse_p4([p4, functional.interpolate(p5, size=p4.shape[2:], mode="nearest")])
        )
        p3_td = self.td_conv_p3(
            self.td_fuse_p3([p3, functional.interpolate(p4_td, size=p3.shape[2:], mode="nearest")])
        )
        p2_out = self.td_conv_p2(
            self.td_fuse_p2([p2, functional.interpolate(p3_td, size=p2.shape[2:], mode="nearest")])
        )

        # Bottom-up pathway
        p3_out = self.bu_conv_p3(
            self.bu_fuse_p3([
                p3,
                p3_td,
                functional.interpolate(p2_out, size=p3.shape[2:], mode="nearest"),
            ])
        )
        p4_out = self.bu_conv_p4(
            self.bu_fuse_p4([
                p4,
                p4_td,
                functional.interpolate(p3_out, size=p4.shape[2:], mode="nearest"),
            ])
        )
        p5_out = self.bu_conv_p5(
            self.bu_fuse_p5([
                p5,
                functional.interpolate(p4_out, size=p5.shape[2:], mode="nearest"),
            ])
        )

        return [p2_out, p3_out, p4_out, p5_out]


# ---------------------------------------------------------------------------
# Detection Head
# ---------------------------------------------------------------------------


class Detect(nn.Module):
    """YOLOv5-style detection head for multiple scales.

    Each scale outputs [batch, num_anchors, grid_h, grid_w, 5 + num_classes].
    """

    stride: torch.Tensor  # computed strides

    def __init__(
        self,
        num_classes: int = 4,
        anchors: list[list[tuple[int, int]]] | None = None,
        channels: list[int] | None = None,
    ):
        super().__init__()
        if anchors is None:
            # Default anchors for 4 scales (P2, P3, P4, P5)
            # These should be auto-computed via k-means on the actual dataset
            anchors = [
                [(5, 4), (7, 8), (12, 10)],        # P2 -- tiny targets
                [(10, 13), (16, 30), (33, 23)],     # P3
                [(30, 61), (62, 45), (59, 119)],    # P4
                [(116, 90), (156, 198), (373, 326)], # P5
            ]
        if channels is None:
            channels = [64, 128, 256, 512]

        self.num_classes = num_classes
        self.num_outputs = 5 + num_classes  # x, y, w, h, obj + cls
        self.num_layers = len(anchors)
        self.num_anchors = len(anchors[0])

        # Register anchors
        a = torch.tensor(anchors, dtype=torch.float32)
        self.register_buffer("anchors", a.view(self.num_layers, -1, 2))
        self.register_buffer(
            "anchor_grid",
            a.clone().view(self.num_layers, 1, -1, 1, 1, 2),
        )

        # Detection convolutions (1x1 conv per scale)
        self.m = nn.ModuleList(
            nn.Conv2d(ch, self.num_outputs * self.num_anchors, 1)
            for ch in channels
        )

        self._init_stride()

    def _init_stride(self) -> None:
        """Initialize stride tensor for 4 detection scales."""
        self.register_buffer(
            "stride",
            torch.tensor([4.0, 8.0, 16.0, 32.0]),
        )

    def forward(
        self, features: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            features: [P2, P3, P4, P5] from neck

        Returns:
            List of detection tensors per scale,
            each [batch, num_anchors, grid_h, grid_w, 5+num_classes]
        """
        outputs = []
        for feat, conv in zip(features, self.m, strict=False):
            bs, _, ny, nx = feat.shape
            x = conv(feat)
            x = x.view(bs, self.num_anchors, self.num_outputs, ny, nx)
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            outputs.append(x)
        return outputs

    def decode(
        self,
        predictions: list[torch.Tensor],
        img_size: int = 640,
    ) -> torch.Tensor:
        """Decode raw predictions to [x, y, w, h, obj, cls...] in pixel coords.

        Used during inference (not training).
        """
        decoded = []
        for i, pred in enumerate(predictions):
            bs, na, ny, nx, no = pred.shape
            device = pred.device

            # Grid offsets
            yv, xv = torch.meshgrid(
                torch.arange(ny, device=device, dtype=torch.float32),
                torch.arange(nx, device=device, dtype=torch.float32),
                indexing="ij",
            )
            grid = torch.stack((xv, yv), dim=2).view(1, 1, ny, nx, 2)

            stride = self.stride[i]
            anchor = self.anchor_grid[i]

            # Decode xy (sigmoid + grid offset) * stride
            xy = (pred[..., :2].sigmoid() * 2.0 - 0.5 + grid) * stride
            # Decode wh (sigmoid * 2)^2 * anchor
            wh = (pred[..., 2:4].sigmoid() * 2.0) ** 2 * anchor
            # Objectness and class probabilities
            conf = pred[..., 4:].sigmoid()

            out = torch.cat((xy, wh, conf), dim=-1)
            decoded.append(out.view(bs, -1, no))

        return torch.cat(decoded, dim=1)


# ---------------------------------------------------------------------------
# YOLOatr -- Full model
# ---------------------------------------------------------------------------


class YOLOatr(nn.Module):
    """YOLOatr: Modified YOLOv5s for thermal IR ATR.

    Architecture:
        Backbone: CSPDarknet53-Small with Focus stem
        Neck: BiFPN (weighted bi-directional FPN)
        Head: 4-scale detection (P2, P3, P4, P5)

    Paper: arxiv 2507.11267
    Params: ~7.1M | GFLOPs: ~16.4
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 3,
        anchors: list[list[tuple[int, int]]] | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Channel widths for YOLOv5s-small (standard)
        w = [32, 64, 128, 256, 512]

        # ---- Backbone: CSPDarknet53-Small ----
        self.focus = Focus(in_channels, w[0], k=3)  # stride 2: 320x320

        # Stage 1: stride 4 -> P2
        self.stage1 = nn.Sequential(
            CBS(w[0], w[1], 3, 2),  # stride 4: 160x160
            C3(w[1], w[1], n=1),
        )

        # Stage 2: stride 8 -> P3
        self.stage2 = nn.Sequential(
            CBS(w[1], w[2], 3, 2),  # stride 8: 80x80
            C3(w[2], w[2], n=3),
        )

        # Stage 3: stride 16 -> P4
        self.stage3 = nn.Sequential(
            CBS(w[2], w[3], 3, 2),  # stride 16: 40x40
            C3(w[3], w[3], n=3),
        )

        # Stage 4: stride 32 -> P5
        self.stage4 = nn.Sequential(
            CBS(w[3], w[4], 3, 2),  # stride 32: 20x20
            C3(w[4], w[4], n=1),
            SPPF(w[4], w[4]),
        )

        # ---- Neck: BiFPN ----
        # Input channels from backbone: P2=64, P3=128, P4=256, P5=512
        # Two BiFPN layers for richer feature fusion, 160 unified channels
        bifpn_channels = 160
        self.bifpn1 = BiFPNBlock(
            channels=[w[1], w[2], w[3], w[4]],
            out_channels=bifpn_channels,
        )
        self.bifpn2 = BiFPNBlock(
            channels=[bifpn_channels] * 4,
            out_channels=bifpn_channels,
        )

        # ---- Head: 4-scale detection ----
        head_channels = [bifpn_channels] * 4  # all 160 after BiFPN
        self.detect = Detect(
            num_classes=num_classes,
            anchors=anchors,
            channels=head_channels,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # Initialize detection head bias for objectness
        for conv in self.detect.m:
            b = conv.bias.view(self.detect.num_anchors, -1)
            # obj bias: log(8 / (640 / stride)^2) -- prior for ~8 objects per image
            b.data[:, 4] += math.log(8 / (640 / 16) ** 2)
            # cls bias: log(1 / (num_classes - 0.99))
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99 + 1e-6))
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def forward(
        self,
        x: torch.Tensor,
        decode: bool = False,
    ) -> list[torch.Tensor] | torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, 640, 640]
            decode: If True, decode predictions to pixel coords (inference)

        Returns:
            Training: list of raw predictions per scale
            Inference (decode=True): decoded detections [B, N, 5+num_classes]
        """
        # Backbone
        x0 = self.focus(x)       # stride 2: [B, 32, 320, 320]
        p2 = self.stage1(x0)     # stride 4: [B, 64, 160, 160]
        p3 = self.stage2(p2)     # stride 8: [B, 128, 80, 80]
        p4 = self.stage3(p3)     # stride 16: [B, 256, 40, 40]
        p5 = self.stage4(p4)     # stride 32: [B, 512, 20, 20]

        # Neck (BiFPN x2)
        fused = self.bifpn1([p2, p3, p4, p5])
        fused = self.bifpn2(fused)  # [P2', P3', P4', P5']

        # Head
        predictions = self.detect(fused)

        if decode:
            return self.detect.decode(predictions)
        return predictions


def build_model(
    num_classes: int = 4,
    in_channels: int = 3,
    anchors: list[list[tuple[int, int]]] | None = None,
) -> YOLOatr:
    """Build YOLOatr model.

    Args:
        num_classes: Number of detection classes (default 4 for DSIAC)
        in_channels: Number of input channels (3 for RGB/IR, 1 for grayscale)
        anchors: Custom anchors per scale, or None for defaults

    Returns:
        YOLOatr model instance
    """
    return YOLOatr(
        num_classes=num_classes,
        in_channels=in_channels,
        anchors=anchors,
    )
