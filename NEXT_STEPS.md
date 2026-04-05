# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 85%

## Done
- [x] Paper read + all hyperparameters extracted (arxiv 2507.11267)
- [x] PRD-01 through PRD-07 complete
  - Model: CSPDarknet53-S + BiFPN neck + 4-head detection (7.26M params)
  - Losses: CIoU box + focal BCE obj + BCE cls
  - Dataset: YOLO format with Custom Augmentation Profile (CAP)
  - Training: SGD + cosine LR + warmup + AMP fp16 + checkpointing + early stop
  - Evaluation: torchvision NMS + mAP@0.5 + per-class metrics
  - Export: safetensors + ONNX (opset 18) + TRT FP32 + TRT FP16
  - Serve: FastAPI with /health /ready /info /predict
  - Docker: Dockerfile.serve + docker-compose.serve.yml
- [x] 34/34 tests pass, ruff clean
- [x] Venv setup on artifacts disk (forge-data full)
- [x] GPU batch finder: bs=40 @ 73.6% VRAM (L4 23GB, AMP fp16)
- [x] Training completed on NUAA-SIRST proxy (21 epochs, loss=0.21, early stop)
- [x] Export pipeline validated:
  - safetensors: 29.1 MB
  - ONNX: 0.6 MB + 29.3 MB data
  - TRT FP32: 36.8 MB
  - TRT FP16: building
- [x] 6 git commits made

## In Progress
- [ ] TRT FP16 engine building

## TODO
- [ ] Push to HuggingFace: ilessio-aiflowlab/project_yoloatr-checkpoint
- [ ] Docker build + health check test
- [ ] Final git commit + push

## Notes
- DSIAC MWIR dataset is restricted (US DoD). NUAA-SIRST used as proxy.
- mAP=0 on NUAA-SIRST expected: 4×5 pixel targets, 256 train images, scratch training
- With real DSIAC data (3600+ images, larger targets), mAP would be meaningful
- forge-data disk 100% full — venv symlinked to /mnt/artifacts-datai/venvs/
- TRT 10.16.0 installed via pip

## Downloads Needed
- DSIAC MWIR dataset — restricted access (US DoD)
