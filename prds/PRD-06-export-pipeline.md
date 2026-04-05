# PRD-06: Export Pipeline

> Status: TODO
> Module: anima_yoloatr

## Objective
Export trained YOLOatr model to ONNX, TensorRT (fp16/fp32), and safetensors formats.

## Export Formats
1. **safetensors** -- portable weight format
2. **ONNX** -- cross-platform inference (opset 17)
3. **TensorRT FP32** -- NVIDIA optimized inference
4. **TensorRT FP16** -- NVIDIA optimized inference (half precision)

## Export Pipeline
```
best.pth -> safetensors -> ONNX -> TRT FP32 + TRT FP16
```

## Tools
- Use shared TRT export toolkit: `/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py`
- ONNX export via torch.onnx.export
- safetensors via safetensors library

## Deliverables
- [ ] src/anima_yoloatr/export.py -- export functions
- [ ] scripts/export.py -- CLI entry point
- [ ] ONNX validation (onnxruntime inference matches PyTorch)
- [ ] TRT fp16 + fp32 builds

## Output Paths
- /mnt/artifacts-datai/exports/project_yoloatr/yoloatr.safetensors
- /mnt/artifacts-datai/exports/project_yoloatr/yoloatr.onnx
- /mnt/artifacts-datai/exports/project_yoloatr/yoloatr_fp32.engine
- /mnt/artifacts-datai/exports/project_yoloatr/yoloatr_fp16.engine

## Acceptance Criteria
- ONNX model loads and runs in onnxruntime
- TRT engines build without errors
- Output difference between PyTorch and ONNX < 1e-4
- FP16 TRT inference < 5ms on L4
