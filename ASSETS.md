# ASSETS.md -- YOLOatr Asset Inventory

> Last updated: 2026-04-05

## Datasets

### DSIAC MWIR ATR Dataset (PRIMARY)
- **Description**: US Army NVESD ATR Algorithm Development Image Database
  - MWIR thermal imagery of ground vehicles in desert environment
  - 10 vehicle classes + 2 human classes; paper uses 4 vehicle classes
  - Ranges: 1000m-5000m, 72 aspect angles, day+night
- **Size**: ~207 GB (full MWIR), subset used ~10-15 GB
- **Format**: ARF files (converted to JPEG frames), YOLO-format annotations
- **Source**: https://dsiac.org/databases/ (US DoD, restricted access)
- **Local path**: /mnt/forge-data/datasets/dsiac_mwir/ (TO BE ACQUIRED)
- **Status**: NOT ON DISK -- requires acquisition from DSIAC

### Alternative / Proxy Datasets for Development

Since DSIAC is restricted, these can serve as development proxies:

| Dataset | Path | Status | Use |
|---------|------|--------|-----|
| NUAA-SIRST | /mnt/forge-data/datasets/nuaa_sirst_yolo/ | ON DISK | IR small target detection dev |
| COCO (val+train) | /mnt/forge-data/datasets/coco/ | ON DISK | General detection baseline |

## Pretrained Models

### Required
| Model | Path | Status | Notes |
|-------|------|--------|-------|
| YOLOv5s weights | N/A | NOT NEEDED | Paper trains from scratch |

### Available on Disk (for reference/comparison)
| Model | Path | Status |
|-------|------|--------|
| YOLOv5l6 | /mnt/forge-data/models/yolov5l6.pt | ON DISK |
| YOLO11n | /mnt/forge-data/models/yolo11n.pt | ON DISK |
| YOLOv12n | /mnt/forge-data/models/yolov12n.pt | ON DISK |

## Shared Infrastructure

### CUDA Kernels (from shared_infra)
| Kernel | Path | Relevance |
|--------|------|-----------|
| Fused image preprocess | cuda_extensions/fused_image_preprocess/ | Input normalization |
| Detection ops | cuda_extensions/detection_ops/ | NMS, IoU computation |
| Vectorized NMS | cuda_extensions/ | Post-processing |

Install:
```bash
uv pip install /mnt/forge-data/shared_infra/cuda_extensions/wheels_py311_cu128/*.whl
```

### Pre-Computed Caches
| Cache | Path | Relevance |
|-------|------|-----------|
| COCO DINOv2 | shared_infra/datasets/coco_dinov2_features/ | Not directly used |

Note: YOLOatr does NOT use DINOv2 or other foundation model features.
It is a standalone YOLOv5-based detector trained from scratch.

## Output Paths

| Type | Path |
|------|------|
| Checkpoints | /mnt/artifacts-datai/checkpoints/project_yoloatr/ |
| Logs | /mnt/artifacts-datai/logs/project_yoloatr/ |
| TensorBoard | /mnt/artifacts-datai/tensorboard/project_yoloatr/ |
| Exports | /mnt/artifacts-datai/exports/project_yoloatr/ |
| Reports | /mnt/artifacts-datai/reports/project_yoloatr/ |

## Downloads Needed

1. **DSIAC MWIR dataset** -- restricted, requires US DoD approval
   - Alternative: use NUAA-SIRST or synthesize thermal-like data from COCO
   - Size: ~207 GB full, ~15 GB for 4-class subset
   - Format: ARF -> JPEG frames + YOLO annotations

2. **No model downloads needed** -- trains from scratch

## Hardware Requirements

| Resource | Paper | Our Setup |
|----------|-------|-----------|
| GPU | Tesla P100 16GB | NVIDIA L4 23GB |
| RAM | 32 GB | Available |
| VRAM needed | ~8-10 GB (batch=32, 640x640) | Fits 1x L4 |
| Training time | ~2-3 hours (100 epochs) | Similar on L4 |
