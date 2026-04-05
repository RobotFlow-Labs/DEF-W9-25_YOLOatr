"""YOLOatr loss functions.

Loss components (YOLOv5-style):
1. Box regression: CIoU loss
2. Objectness: BCE with focal loss (gamma=0.3)
3. Classification: BCE loss

Paper: arxiv 2507.11267
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    giou: bool = False,
    ciou: bool = True,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute IoU, GIoU, or CIoU between two sets of boxes.

    Args:
        box1: [N, 4] predicted boxes
        box2: [N, 4] target boxes
        xywh: If True, boxes are in (cx, cy, w, h) format
        giou: Compute GIoU
        ciou: Compute CIoU (takes precedence over giou)
        eps: Numerical stability

    Returns:
        IoU/GIoU/CIoU tensor [N]
    """
    if xywh:
        # Convert xywh -> xyxy
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2
        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = (
            box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        )
        b2_x1, b2_y1, b2_x2, b2_y2 = (
            box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
        )

    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union
    w1 = b1_x2 - b1_x1
    h1 = b1_y2 - b1_y1
    w2 = b2_x2 - b2_x1
    h2 = b2_y2 - b2_y1
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter + eps

    iou = inter / union

    if giou or ciou:
        # Smallest enclosing box
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

        if ciou:
            c2 = cw**2 + ch**2 + eps  # diagonal squared
            # Center distance squared
            rho2 = (
                (box1[..., 0] - box2[..., 0]) ** 2
                + (box1[..., 1] - box2[..., 1]) ** 2
                if xywh
                else (
                    ((b1_x1 + b1_x2) - (b2_x1 + b2_x2)) ** 2
                    + ((b1_y1 + b1_y2) - (b2_y1 + b2_y2)) ** 2
                )
                / 4
            )
            # Aspect ratio consistency
            import math

            v = (4 / math.pi**2) * (
                torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))
            ) ** 2
            with torch.no_grad():
                alpha = v / (1 - iou + v + eps)
            return iou - rho2 / c2 - alpha * v
        else:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area

    return iou


class FocalBCELoss(nn.Module):
    """Binary cross-entropy with focal loss modifier.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Default gamma=0.3 from YOLOatr custom augmentation profile.
    """

    def __init__(self, gamma: float = 0.3, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal BCE loss.

        Args:
            pred: Predictions (logits, not sigmoid)
            target: Binary targets

        Returns:
            Focal loss scalar
        """
        bce = functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = torch.exp(-bce)  # probability of correct class
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha weighting
        alpha_factor = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)

        loss = alpha_factor * focal_weight * bce
        return loss.mean()


class ComputeLoss:
    """YOLOv5-style loss computation for YOLOatr.

    Handles anchor matching, target building, and multi-scale loss aggregation
    for 4 detection scales (P2, P3, P4, P5).
    """

    def __init__(
        self,
        model: nn.Module,
        box_gain: float = 0.05,
        obj_gain: float = 1.0,
        cls_gain: float = 0.5,
        focal_gamma: float = 0.3,
        anchor_threshold: float = 4.0,
        label_smoothing: float = 0.0,
    ):
        self.model = model
        self.box_gain = box_gain
        self.obj_gain = obj_gain
        self.cls_gain = cls_gain
        self.anchor_threshold = anchor_threshold
        self.num_classes = model.num_classes

        # Loss functions
        self.bce_obj = FocalBCELoss(gamma=focal_gamma)
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="mean")

        # Label smoothing
        self.cp = 1.0 - 0.5 * label_smoothing  # positive label
        self.cn = 0.5 * label_smoothing  # negative label

        # Objectness loss balance per layer (P2, P3, P4, P5)
        self.balance = [4.0, 1.0, 0.4, 0.1]

        # Get anchors and strides from model
        detect = model.detect
        self.anchors = detect.anchors  # [num_layers, num_anchors, 2]
        self.strides = detect.stride  # [num_layers]
        self.num_anchors = detect.num_anchors
        self.num_layers = detect.num_layers

    def build_targets(
        self,
        predictions: list[torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[list, list, list, list]:
        """Build training targets for all detection scales.

        Args:
            predictions: List of predictions per scale
            targets: [num_targets, 6] -- (batch_idx, class, cx, cy, w, h)
                     coordinates normalized to [0, 1]

        Returns:
            Tuple of (target_cls, target_box, indices, anchors) per layer
        """
        device = targets.device
        num_targets = targets.shape[0]

        tcls_all, tbox_all, indices_all, anch_all = [], [], [], []

        gain = torch.ones(6, device=device)

        for i in range(self.num_layers):
            anchors = self.anchors[i]  # [na, 2]
            na = anchors.shape[0]
            _, _, ny, nx, _ = predictions[i].shape

            gain[2:6] = torch.tensor(
                [nx, ny, nx, ny], device=device, dtype=torch.float32
            )

            if num_targets > 0:
                # Match targets to anchors by aspect ratio
                # Expand targets for each anchor
                t_expanded = targets.unsqueeze(1).repeat(1, na, 1)  # [nt, na, 6]
                t_scaled = t_expanded.clone()
                t_scaled[..., 2:6] *= gain[2:6]

                # Compute ratio between target wh and anchor wh
                r = t_scaled[..., 4:6] / anchors[None, :, :]  # [nt, na, 2]
                # Filter by max ratio threshold
                mask = torch.max(r, 1.0 / r).max(dim=2).values < self.anchor_threshold
                # [nt, na]

                # Select matched targets
                t_matched = t_scaled[mask]  # [N_matched, 6]
                anchor_idx = (
                    torch.arange(na, device=device)
                    .unsqueeze(0)
                    .repeat(num_targets, 1)[mask]
                )

                if t_matched.shape[0] > 0:
                    b_idx = t_matched[:, 0].long()
                    cls = t_matched[:, 1].long()
                    gxy = t_matched[:, 2:4]  # grid xy
                    gwh = t_matched[:, 4:6]  # grid wh

                    gij = gxy.long()
                    gi = gij[:, 0].clamp(0, nx - 1)
                    gj = gij[:, 1].clamp(0, ny - 1)

                    indices_all.append((b_idx, anchor_idx, gj, gi))
                    tbox_all.append(
                        torch.cat((gxy - gij.float(), gwh), dim=1)
                    )
                    anch_all.append(anchors[anchor_idx])
                    tcls_all.append(cls)
                else:
                    indices_all.append(
                        (
                            torch.zeros(0, device=device, dtype=torch.long),
                            torch.zeros(0, device=device, dtype=torch.long),
                            torch.zeros(0, device=device, dtype=torch.long),
                            torch.zeros(0, device=device, dtype=torch.long),
                        )
                    )
                    tbox_all.append(torch.zeros(0, 4, device=device))
                    anch_all.append(torch.zeros(0, 2, device=device))
                    tcls_all.append(torch.zeros(0, device=device, dtype=torch.long))
            else:
                indices_all.append(
                    (
                        torch.zeros(0, device=device, dtype=torch.long),
                        torch.zeros(0, device=device, dtype=torch.long),
                        torch.zeros(0, device=device, dtype=torch.long),
                        torch.zeros(0, device=device, dtype=torch.long),
                    )
                )
                tbox_all.append(torch.zeros(0, 4, device=device))
                anch_all.append(torch.zeros(0, 2, device=device))
                tcls_all.append(torch.zeros(0, device=device, dtype=torch.long))

        return tcls_all, tbox_all, indices_all, anch_all

    def __call__(
        self,
        predictions: list[torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total loss.

        Args:
            predictions: List of raw predictions per scale
                Each: [B, num_anchors, grid_h, grid_w, 5+num_classes]
            targets: [num_targets, 6] -- (batch_idx, class, cx, cy, w, h)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        device = predictions[0].device
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        loss_cls = torch.zeros(1, device=device)

        tcls, tbox, indices, anchors = self.build_targets(predictions, targets)

        for i, pred in enumerate(predictions):
            b_idx, a_idx, gj, gi = indices[i]
            n_targets = b_idx.shape[0]

            # Objectness target (all zeros initially)
            tobj = torch.zeros(pred.shape[:4], dtype=pred.dtype, device=device)

            if n_targets > 0:
                # Select predictions for matched targets
                ps = pred[b_idx, a_idx, gj, gi]  # [N, 5+nc]

                # Box loss (CIoU)
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2.0) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), dim=1)
                iou = bbox_iou(pbox, tbox[i], xywh=True, ciou=True)
                loss_box += (1.0 - iou).mean()

                # Set objectness target to IoU
                tobj[b_idx, a_idx, gj, gi] = iou.detach().clamp(0).to(tobj.dtype)

                # Classification loss
                if self.num_classes > 1:
                    t_cls = torch.full_like(
                        ps[:, 5:], self.cn, device=device
                    )
                    t_cls[range(n_targets), tcls[i]] = self.cp
                    loss_cls += self.bce_cls(ps[:, 5:], t_cls)

            # Objectness loss (per layer, balanced)
            loss_obj += self.bce_obj(pred[..., 4], tobj) * self.balance[i]

        # Apply gains
        loss_box *= self.box_gain
        loss_obj *= self.obj_gain
        loss_cls *= self.cls_gain

        total = loss_box + loss_obj + loss_cls

        loss_dict = {
            "box": loss_box.item(),
            "obj": loss_obj.item(),
            "cls": loss_cls.item(),
            "total": total.item(),
        }

        return total.squeeze(), loss_dict
