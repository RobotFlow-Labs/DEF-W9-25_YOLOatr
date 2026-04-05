# Tasks Index -- YOLOatr

> Last updated: 2026-04-05

## PRD-01: Foundation
- [x] T01.01: Create CLAUDE.md with paper summary
- [x] T01.02: Create ASSETS.md with dataset/model inventory
- [x] T01.03: Create PRD.md with 7-PRD build plan table
- [x] T01.04: Create 7 PRD files in prds/
- [x] T01.05: Create tasks/INDEX.md
- [x] T01.06: Create NEXT_STEPS.md
- [x] T01.07: Create anima_module.yaml
- [x] T01.08: Create pyproject.toml (hatchling)
- [x] T01.09: Create configs/paper.toml and configs/debug.toml
- [x] T01.10: Create src/anima_yoloatr/__init__.py
- [x] T01.11: Create src/anima_yoloatr/model.py (YOLOatr architecture)
- [x] T01.12: Create src/anima_yoloatr/losses.py (CIoU, focal BCE)
- [x] T01.13: Create src/anima_yoloatr/dataset.py (YOLO format + CAP)
- [x] T01.14: Create src/anima_yoloatr/train.py (training loop)
- [x] T01.15: Create src/anima_yoloatr/evaluate.py (mAP evaluation)
- [x] T01.16: Create src/anima_yoloatr/utils.py (utilities)
- [x] T01.17: Create scripts/train.py and scripts/evaluate.py
- [x] T01.18: Create tests/test_model.py and tests/test_dataset.py
- [x] T01.19: Create Dockerfile.serve and docker-compose.serve.yml

## PRD-02: Core Model
- [ ] T02.01: Implement CBS (Conv-BN-SiLU) block
- [ ] T02.02: Implement Bottleneck and C3 (CSP Bottleneck x3) blocks
- [ ] T02.03: Implement SPPF (Spatial Pyramid Pooling Fast)
- [ ] T02.04: Implement CSPDarknet53-Small backbone with P2 output
- [ ] T02.05: Implement BiFPN neck with weighted fusion
- [ ] T02.06: Implement 4-scale detection head (P2/P3/P4/P5)
- [ ] T02.07: Implement auto-anchor computation (k-means)
- [ ] T02.08: Verify parameter count (~7.1M) and GFLOPs (~16.4)
- [ ] T02.09: Unit test forward pass with random input

## PRD-03: Loss Functions
- [ ] T03.01: Implement IoU computation (IoU, GIoU, CIoU)
- [ ] T03.02: Implement Focal Loss with configurable gamma
- [ ] T03.03: Implement BCE objectness loss with layer scaling
- [ ] T03.04: Implement BCE classification loss
- [ ] T03.05: Implement build_targets for 4-head anchor matching
- [ ] T03.06: Implement ComputeLoss aggregator class
- [ ] T03.07: Unit test loss backward pass

## PRD-04: Training Pipeline
- [ ] T04.01: Implement YOLO-format dataset loader
- [ ] T04.02: Implement letterbox resize and padding
- [ ] T04.03: Implement mosaic augmentation (prob=0.1)
- [ ] T04.04: Implement mixup augmentation (prob=0.4)
- [ ] T04.05: Implement copy-paste augmentation (prob=0.5)
- [ ] T04.06: Implement HSV augmentation (h=0.015, s=0.7, v=0.4)
- [ ] T04.07: Implement geometric augmentations (rotate, translate, scale, flip)
- [ ] T04.08: Implement SGD optimizer with cosine scheduler + warmup
- [ ] T04.09: Implement training loop with mixed precision
- [ ] T04.10: Implement checkpoint manager (top-2 by val_mAP)
- [ ] T04.11: Implement early stopping (patience=10)
- [ ] T04.12: Implement NaN detection and gradient clipping
- [ ] T04.13: Smoke test: 2 epochs on synthetic data

## PRD-05: Evaluation
- [ ] T05.01: Implement NMS post-processing
- [ ] T05.02: Implement precision/recall computation
- [ ] T05.03: Implement AP (area under PR curve) per class
- [ ] T05.04: Implement mAP@0.5 aggregation
- [ ] T05.05: Implement confusion matrix
- [ ] T05.06: Implement PR curve plotting
- [ ] T05.07: Implement correlated vs decorrelated protocol split
- [ ] T05.08: Generate evaluation report markdown

## PRD-06: Export Pipeline
- [ ] T06.01: Export to safetensors
- [ ] T06.02: Export to ONNX (opset 17, dynamic batch)
- [ ] T06.03: Validate ONNX with onnxruntime
- [ ] T06.04: Export to TensorRT FP32
- [ ] T06.05: Export to TensorRT FP16
- [ ] T06.06: Benchmark inference latency across formats

## PRD-07: Integration
- [ ] T07.01: Create AnimaNode subclass (serve.py)
- [ ] T07.02: Build and test Docker image
- [ ] T07.03: Verify /health, /ready, /predict endpoints
- [ ] T07.04: Create ROS2 detection publisher
- [ ] T07.05: Push model + weights to HuggingFace
- [ ] T07.06: Create HF model card
