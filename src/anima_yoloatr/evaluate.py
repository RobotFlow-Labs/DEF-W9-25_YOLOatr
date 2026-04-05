"""YOLOatr evaluation pipeline.

Metrics:
- Precision, Recall, F1
- AP per class (area under PR curve)
- mAP@0.5

Supports correlated (T1) and decorrelated (T2) test protocols.

Paper: arxiv 2507.11267
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from anima_yoloatr.utils import load_config


def non_max_suppression(
    predictions: torch.Tensor,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.6,
    max_detections: int = 300,
) -> list[torch.Tensor]:
    """Apply NMS to decoded predictions.

    Args:
        predictions: [B, N, 5+nc] decoded predictions (x, y, w, h, obj, cls...)
        conf_threshold: Minimum confidence threshold
        iou_threshold: NMS IoU threshold
        max_detections: Maximum detections per image

    Returns:
        List of [M, 6] tensors per image: (x1, y1, x2, y2, conf, class)
    """
    batch_size = predictions.shape[0]
    outputs = []

    for bi in range(batch_size):
        pred = predictions[bi]  # [N, 5+nc]

        # Filter by objectness
        obj_conf = pred[:, 4]
        mask = obj_conf > conf_threshold
        pred = pred[mask]

        if pred.shape[0] == 0:
            outputs.append(torch.zeros((0, 6), device=predictions.device))
            continue

        # Compute class confidence = obj * cls
        cls_conf, cls_idx = pred[:, 5:].max(dim=1, keepdim=True)
        conf = pred[:, 4:5] * cls_conf

        # Filter by combined confidence
        mask = conf.squeeze() > conf_threshold
        pred = pred[mask]
        conf = conf[mask]
        cls_idx = cls_idx[mask]

        if pred.shape[0] == 0:
            outputs.append(torch.zeros((0, 6), device=predictions.device))
            continue

        # Convert xywh -> xyxy
        x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        # NMS using torchvision (fast, batched)
        conf_flat = conf.squeeze(-1)
        cls_flat = cls_idx.squeeze(-1).float()

        # Offset boxes by class for batched NMS
        keep = torchvision.ops.batched_nms(
            boxes, conf_flat, cls_flat.int(), iou_threshold,
        )
        keep = keep[:max_detections]

        if keep.numel() > 0:
            result = torch.cat([
                boxes[keep],
                conf_flat[keep].unsqueeze(-1),
                cls_flat[keep].unsqueeze(-1),
            ], dim=1)
            outputs.append(result)
        else:
            outputs.append(torch.zeros((0, 6), device=predictions.device))

    return outputs


def _nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> list[int]:
    """Simple NMS implementation."""
    if boxes.shape[0] == 0:
        return []

    # Sort by score
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break

        i = order[0].item()
        keep.append(i)

        # Compute IoU of top box with remaining
        xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
        yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
        xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
        yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])

        inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (
            (boxes[order[1:], 2] - boxes[order[1:], 0])
            * (boxes[order[1:], 3] - boxes[order[1:], 1])
        )
        iou = inter / (area_i + area_j - inter + 1e-7)

        remaining = (iou <= iou_threshold).nonzero(as_tuple=True)[0]
        order = order[remaining + 1]

    return keep


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP from precision-recall curve (COCO-style, all-point interpolation)."""
    # Prepend sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (Delta recall) * precision
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def compute_metrics(
    all_detections: list[torch.Tensor],
    all_targets: list[torch.Tensor],
    num_classes: int = 4,
    iou_threshold: float = 0.5,
) -> dict:
    """Compute precision, recall, AP per class, and mAP@0.5.

    Args:
        all_detections: List of [M, 6] tensors (x1, y1, x2, y2, conf, cls)
        all_targets: List of [K, 5] tensors (cls, cx, cy, w, h) normalized
        num_classes: Number of classes
        iou_threshold: IoU threshold for TP/FP

    Returns:
        Dict with precision, recall, ap per class, mAP@0.5
    """
    # Collect all predictions and ground truths
    all_tp = {c: [] for c in range(num_classes)}
    all_conf = {c: [] for c in range(num_classes)}
    num_gt = dict.fromkeys(range(num_classes), 0)

    for dets, gts in zip(all_detections, all_targets, strict=False):
        if gts.shape[0] > 0:
            for c in range(num_classes):
                gt_mask = gts[:, 0] == c
                num_gt[c] += gt_mask.sum().item()

        if dets.shape[0] == 0:
            continue

        gt_matched = torch.zeros(gts.shape[0], dtype=torch.bool)

        for det in dets:
            cls_id = int(det[5].item())
            conf = det[4].item()
            all_conf[cls_id].append(conf)

            if gts.shape[0] == 0:
                all_tp[cls_id].append(0)
                continue

            # Find best matching GT of same class
            gt_cls_mask = (gts[:, 0] == cls_id) & (~gt_matched)
            if not gt_cls_mask.any():
                all_tp[cls_id].append(0)
                continue

            # Convert GT xywh to xyxy (assuming pixel coords after scaling)
            gt_boxes = gts[gt_cls_mask]
            gt_x1 = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
            gt_y1 = gt_boxes[:, 2] - gt_boxes[:, 4] / 2
            gt_x2 = gt_boxes[:, 1] + gt_boxes[:, 3] / 2
            gt_y2 = gt_boxes[:, 2] + gt_boxes[:, 4] / 2

            # IoU with detection
            dx1, dy1, dx2, dy2 = det[0], det[1], det[2], det[3]
            inter_x1 = torch.max(dx1, gt_x1)
            inter_y1 = torch.max(dy1, gt_y1)
            inter_x2 = torch.min(dx2, gt_x2)
            inter_y2 = torch.min(dy2, gt_y2)
            inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

            area_det = (dx2 - dx1) * (dy2 - dy1)
            area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
            iou = inter / (area_det + area_gt - inter + 1e-7)

            best_iou, best_idx = iou.max(dim=0)
            if best_iou >= iou_threshold:
                # Find the actual index in gts
                gt_indices = gt_cls_mask.nonzero(as_tuple=True)[0]
                actual_idx = gt_indices[best_idx]
                if not gt_matched[actual_idx]:
                    gt_matched[actual_idx] = True
                    all_tp[cls_id].append(1)
                else:
                    all_tp[cls_id].append(0)
            else:
                all_tp[cls_id].append(0)

    # Compute per-class metrics
    results = {}
    aps = []
    for c in range(num_classes):
        if num_gt[c] == 0:
            results[f"class_{c}_ap"] = 0.0
            results[f"class_{c}_precision"] = 0.0
            results[f"class_{c}_recall"] = 0.0
            continue

        tp = np.array(all_tp[c])
        conf = np.array(all_conf[c])

        if len(tp) == 0:
            results[f"class_{c}_ap"] = 0.0
            results[f"class_{c}_precision"] = 0.0
            results[f"class_{c}_recall"] = 0.0
            continue

        # Sort by confidence (descending)
        sort_idx = np.argsort(-conf)
        tp = tp[sort_idx]

        # Cumulative TP and FP
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(1 - tp)

        recall = cum_tp / num_gt[c]
        precision = cum_tp / (cum_tp + cum_fp + 1e-7)

        ap = compute_ap(recall, precision)
        aps.append(ap)

        results[f"class_{c}_ap"] = ap
        results[f"class_{c}_precision"] = float(precision[-1]) if len(precision) > 0 else 0.0
        results[f"class_{c}_recall"] = float(recall[-1]) if len(recall) > 0 else 0.0

    results["map50"] = float(np.mean(aps)) if aps else 0.0
    return results


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.5,
    nms_threshold: float = 0.6,
    num_classes: int = 4,
) -> dict:
    """Evaluate model on a dataset.

    Args:
        model: YOLOatr model
        dataloader: Validation/test dataloader
        device: CUDA/CPU device
        conf_threshold: Confidence threshold for NMS
        iou_threshold: IoU threshold for mAP
        nms_threshold: NMS IoU threshold

    Returns:
        Metrics dict with mAP@0.5, per-class AP, precision, recall
    """
    model.eval()

    all_detections = []
    all_targets = []

    for batch in dataloader:
        images = batch["images"].to(device)
        labels = batch["labels"]  # [N, 6]: batch_idx, cls, cx, cy, w, h

        # Forward pass with decoding
        predictions = model(images, decode=True)

        # NMS
        detections = non_max_suppression(
            predictions,
            conf_threshold=conf_threshold,
            iou_threshold=nms_threshold,
        )

        # Separate labels per image in batch
        bs = images.shape[0]
        img_size = images.shape[2]  # 640
        for bi in range(bs):
            all_detections.append(detections[bi].cpu())
            # Get targets for this image
            mask = labels[:, 0] == bi
            img_targets = labels[mask, 1:].clone()  # [K, 5]: cls, cx, cy, w, h
            # Scale normalized coords to pixel coords to match detections
            if img_targets.shape[0] > 0:
                img_targets[:, 1:] *= img_size
            all_targets.append(img_targets)

    metrics = compute_metrics(
        all_detections, all_targets,
        num_classes=num_classes,
        iou_threshold=iou_threshold,
    )

    return metrics


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate YOLOatr")
    parser.add_argument(
        "--config", type=str, default="configs/paper.toml",
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--protocol", type=str, default="correlated",
        choices=["correlated", "decorrelated"],
        help="Test protocol (T1=correlated, T2=decorrelated)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    from anima_yoloatr.dataset import YOLODataset, collate_fn
    from anima_yoloatr.model import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model_cfg = config.get("model", {})
    model = build_model(num_classes=model_cfg.get("num_classes", 4))
    model = model.to(device)

    # Load weights
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])

    # Dataset
    data_cfg = config.get("data", {})
    data_path = data_cfg.get(f"{args.split}_path", data_cfg.get("test_path", ""))
    dataset = YOLODataset(data_root=data_path, img_size=model_cfg.get("input_size", 640))
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    # Evaluate
    eval_cfg = config.get("evaluation", {})
    metrics = evaluate_model(
        model, loader, device,
        conf_threshold=eval_cfg.get("conf_threshold", 0.001),
        iou_threshold=eval_cfg.get("iou_threshold", 0.5),
        nms_threshold=eval_cfg.get("nms_threshold", 0.6),
    )

    # Print results
    class_names = model_cfg.get("class_names", [f"class_{i}" for i in range(4)])
    print(f"\n{'='*60}")
    print(f"YOLOatr Evaluation Results ({args.protocol} protocol)")
    print(f"{'='*60}")
    print(f"  mAP@0.5: {metrics['map50']:.4f}")
    for i, name in enumerate(class_names):
        print(
            f"  {name}: AP={metrics.get(f'class_{i}_ap', 0):.4f} "
            f"P={metrics.get(f'class_{i}_precision', 0):.4f} "
            f"R={metrics.get(f'class_{i}_recall', 0):.4f}"
        )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
