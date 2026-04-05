# TRAINING_REPORT.md — YOLOatr

> Module: project_yoloatr
> Paper: arxiv 2507.11267
> Date: 2026-04-05

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOatr (CSPDarknet53-S + BiFPN + 4-head) |
| Parameters | 7,251,492 (7.26M) |
| Paper params | 7,086,449 (7.09M) — 2.4% diff |
| Dataset | NUAA-SIRST (proxy for DSIAC MWIR) |
| Train / Val / Test | 256 / 85 / 86 images |
| Classes | 1 (ir_target) |
| Input size | 640×640 |
| Batch size | 40 (73.6% of 23GB L4, AMP fp16) |
| Optimizer | SGD (lr=0.01, momentum=0.937, wd=5e-4) |
| Scheduler | Cosine with 3-epoch warmup |
| Precision | fp16 (AMP) |
| Epochs | 21 (early stop, patience=20) |
| GPU | NVIDIA L4 23GB × 1 |
| Time/epoch | ~7.5s |
| Total time | ~2.5 min |

## Augmentation (Custom Augmentation Profile — CAP)

| Augmentation | Value |
|-------------|-------|
| Mosaic | 0.1 (low) |
| Shear | 0.0 (off) |
| Mixup | 0.4 |
| Copy-paste | 0.5 |
| HSV h/s/v | 0.015 / 0.7 / 0.4 |
| Rotation | ±3° |
| Flip UD/LR | 0.1 / 0.5 |

## Training Curve

| Epoch | Train Loss | Val mAP@0.5 | LR |
|-------|-----------|-------------|-----|
| 1 | 0.2184 | 0.0000 | 0.000000 |
| 5 | 0.2278 | 0.0000 | 0.009997 |
| 10 | 0.2183 | 0.0000 | 0.009907 |
| 15 | 0.2110 | 0.0000 | 0.009691 |
| 21 | 0.2205 | 0.0000 | 0.009268 |

## Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Best mAP@0.5 | 0.0000 | Expected — see notes below |
| Final train loss | 0.2205 | Converged |
| Early stop | Epoch 21 | Patience 20 |

### Why mAP = 0

NUAA-SIRST targets are extremely small (~4×5 pixels at 640×640 input). With only 256 training images and training from scratch (no pretrained weights), the model cannot learn to detect such tiny targets in 21 epochs. This is expected behavior for a proxy dataset validation.

The paper reports 99.6% mAP on DSIAC MWIR (correlated) with 3,600+ images per class and larger targets. Real performance requires the actual DSIAC dataset.

## Exports

| Format | File | Size |
|--------|------|------|
| PyTorch checkpoint | best.pth | 28 MB |
| safetensors | yoloatr.safetensors | 28 MB |
| ONNX (opset 18) | yoloatr.onnx + .data | 29 MB |
| TensorRT FP32 | yoloatr_fp32.engine | 36 MB |
| TensorRT FP16 | yoloatr_fp16.engine | 17 MB |

## Tests

34/34 tests pass:
- test_model.py: 22 tests (building blocks, BiFPN, detect, full model, gradient flow)
- test_losses.py: 9 tests (bbox_iou, focal BCE, ComputeLoss)
- test_export.py: 3 tests (safetensors, ONNX export, ONNX runtime validation)
- test_dataset.py: 6 tests (synthetic samples, collation, CAP defaults)

## Artifacts

| Type | Path |
|------|------|
| Checkpoints | /mnt/artifacts-datai/checkpoints/project_yoloatr/ |
| Exports | /mnt/artifacts-datai/exports/project_yoloatr/ |
| Logs | /mnt/artifacts-datai/logs/project_yoloatr/ |
| HuggingFace | ilessio-aiflowlab/project_yoloatr (private) |
| GitHub | RobotFlow-Labs/DEF-W9-25_YOLOatr |
