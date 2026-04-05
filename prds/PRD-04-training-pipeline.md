# PRD-04: Training Pipeline

> Status: TODO
> Module: anima_yoloatr

## Objective
Implement the full training pipeline: data loading, custom augmentation profile (CAP),
training loop with SGD, checkpointing, and logging.

## Dataset Pipeline

### Data Format
- YOLO format: images/ + labels/ directories
- Labels: class_id x_center y_center width height (normalized)
- 4 classes: T72_Tank(0), BTR70(1), SUV(2), Pickup(3)

### Data Loading
- LetterBox resize to 640x640
- Mosaic augmentation (probability 0.1)
- MixUp augmentation (probability 0.4)
- Copy-paste augmentation (probability 0.5)

### Custom Augmentation Profile (CAP)
| Param | Value | Rationale |
|-------|-------|-----------|
| hsv_h | 0.015 | Minimal hue shift for IR |
| hsv_s | 0.7 | Saturation variation |
| hsv_v | 0.4 | Value/brightness variation |
| degrees | 3 | Slight rotation |
| translate | 0.1 | Medium translation |
| scale | 0.3 | Scale jitter |
| shear | 0.0 | OFF -- counterproductive for tiny IR targets |
| perspective | 0.0005 | Minimal perspective |
| flipud | 0.1 | Low vertical flip |
| fliplr | 0.5 | Horizontal flip |
| mosaic | 0.1 | LOW -- small targets get even smaller |
| mixup | 0.4 | High -- increases diversity |
| copy_paste | 0.5 | High -- cross-stitch augmentation |

## Training Configuration
- Optimizer: SGD (lr=0.01, momentum=0.937, weight_decay=0.0005)
- Scheduler: cosine annealing with linear warmup (first 3 epochs)
- Epochs: 100
- Batch size: 32 (paper) / auto-detect on our hardware
- Precision: fp16 mixed precision
- Gradient clipping: max_norm=1.0
- Training from scratch (random initialization)

## Checkpointing
- Save every 10 epochs + best model
- Keep top 2 by val mAP@0.5
- Resume support
- Output: /mnt/artifacts-datai/checkpoints/project_yoloatr/

## Logging
- Console: [Epoch N/100] train_loss=X val_mAP=Y lr=Z
- TensorBoard: /mnt/artifacts-datai/tensorboard/project_yoloatr/
- JSON metrics: /mnt/artifacts-datai/logs/project_yoloatr/

## Deliverables
- [ ] src/anima_yoloatr/dataset.py -- YOLO format dataset with CAP augmentations
- [ ] src/anima_yoloatr/train.py -- training loop
- [ ] scripts/train.py -- CLI entry point
- [ ] configs/paper.toml -- exact paper hyperparameters
- [ ] configs/debug.toml -- quick smoke test

## Acceptance Criteria
- Dataset loads and augments correctly
- Training loop runs for 2 epochs without error (smoke test)
- Checkpoints save and resume correctly
- Loss decreases over epochs
- GPU utilization 60-80% on L4
