# PRD.md -- YOLOatr Master Build Plan

> Module: anima_yoloatr
> Paper: YOLOatr (arxiv 2507.11267)
> Last updated: 2026-04-05

## Overview

YOLOatr is a modified YOLOv5s for Automatic Target Recognition in MWIR thermal
infrared imagery. Key modifications: BiFPN neck, extra P2 small-object detection head,
and a custom augmentation profile (CAP) for IR domain.

## Build Plan -- 7 PRDs

| PRD | Title | Status | Description |
|-----|-------|--------|-------------|
| PRD-01 | Foundation | DONE | Project scaffolding, configs, pyproject.toml |
| PRD-02 | Core Model | DONE | YOLOatr architecture (7.26M params, 2.4% of paper) |
| PRD-03 | Loss Functions | DONE | CIoU box loss, BCE obj/cls loss, focal loss |
| PRD-04 | Training Pipeline | DONE | Data loading, augmentation (CAP), training loop |
| PRD-05 | Evaluation | DONE | mAP@0.5, precision, recall, per-class metrics |
| PRD-06 | Export Pipeline | DONE | ONNX, TensorRT fp16/fp32, safetensors |
| PRD-07 | Integration | DONE | Docker serve, ROS2 node, HF push |

## Architecture Summary

```
Input (640x640x1 or 3)
    |
CSPDarknet53-Small (Backbone)
    |-- P2 (stride 4, 160x160)
    |-- P3 (stride 8, 80x80)
    |-- P4 (stride 16, 40x40)
    |-- P5 (stride 32, 20x20)
    |
BiFPN Neck (weighted bi-directional FPN)
    |-- Fused P2, P3, P4, P5
    |
4x Detection Heads
    |-- P2: small objects (extra head)
    |-- P3: small-medium objects
    |-- P4: medium objects
    |-- P5: large objects
    |
NMS -> Detections [x, y, w, h, conf, class]
```

## Key Design Decisions

1. **From scratch training** -- no ImageNet pretrain (paper shows scratch > transfer for IR)
2. **4 detection heads** -- P2 added for tiny targets at long range
3. **BiFPN** -- weighted feature fusion replaces PANet
4. **Custom augmentation** -- low mosaic, no shear, high mixup/copy-paste for IR domain
5. **SGD optimizer** -- better generalization than Adam per paper findings
6. **Anchor-based** -- k-means auto-anchors, 3 anchors per detection scale
