# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 75%

## Done
- [x] Read paper (arxiv 2507.11267) -- extracted all architecture, hyperparameters, metrics
- [x] PRD-01 Foundation: complete scaffolding built
  - CLAUDE.md, ASSETS.md, PRD.md, 7 PRD files
  - tasks/INDEX.md with granular tasks per PRD
  - anima_module.yaml, pyproject.toml
  - configs/paper.toml, configs/debug.toml
  - src/anima_yoloatr/ package (model, losses, dataset, train, evaluate, utils)
  - scripts/train.py, scripts/evaluate.py
  - tests/test_model.py, tests/test_dataset.py
  - Dockerfile.serve, docker-compose.serve.yml
- [x] PRD-02 Core Model: YOLOatr architecture verified
  - CSPDarknet53-Small backbone + BiFPN neck + 4 detection heads
  - Parameter count: 7,257,288 (2.4% within paper's 7,086,449)
  - All building blocks tested (CBS, C3, SPPF, Focus, BiFPN, Detect)
  - Forward pass verified (training + inference mode)
- [x] PRD-03 Loss Functions: CIoU + focal BCE + target building
  - bbox_iou (IoU/GIoU/CIoU), FocalBCELoss, ComputeLoss
  - Multi-scale loss aggregation for 4 detection heads
  - Anchor matching with aspect ratio threshold
  - Gradient flow verified through loss computation
- [x] PRD-04 Training Pipeline: complete training loop
  - SGD optimizer with cosine LR schedule + warmup
  - Mixed precision (fp16), gradient clipping
  - Checkpoint management (top-2 by val_mAP)
  - Early stopping, NaN detection
  - Resume from checkpoint support
  - TensorBoard logging
- [x] PRD-05 Evaluation: NMS + AP computation
  - Non-max suppression with per-class processing
  - mAP@0.5, precision, recall, per-class AP
  - Correlated/decorrelated test protocols
- [x] PRD-06 Export Pipeline: ONNX + safetensors + TRT
  - export.py with full pipeline: pth -> safetensors -> ONNX -> TRT
  - ONNX validation vs PyTorch output
  - TRT fp16/fp32 via trtexec + shared toolkit fallback
  - CLI entry point: yoloatr-export
- [x] PRD-07 Integration: Docker + serve + API
  - serve.py with FastAPI (health/ready/info/predict)
  - Dockerfile.serve, docker-compose.serve.yml
  - ROS2 node config in anima_module.yaml
- [x] NUAA-SIRST proxy config for development
  - configs/nuaa_sirst.toml (1-class IR target detection)
  - 256 train, 85 val, 86 test images verified
- [x] 34/34 tests pass, ruff clean

## In Progress
- [ ] Nothing currently in progress

## TODO
- [ ] GPU training on NUAA-SIRST (need GPU availability)
- [ ] Run /gpu-batch-finder for optimal batch size
- [ ] Full training run with nohup+disown
- [ ] Export trained model (ONNX + TRT fp16/fp32)
- [ ] Push to HuggingFace: ilessio-aiflowlab/project_yoloatr-checkpoint
- [ ] Docker build + health check

## Blocking
- DSIAC MWIR dataset is restricted (US DoD). Using NUAA-SIRST as proxy for development.
- forge-data disk 100% full — venv symlinked to /mnt/artifacts-datai/

## Downloads Needed
- DSIAC MWIR dataset -- ~207 GB -- restricted access (not available)
- No model downloads needed (trains from scratch)
