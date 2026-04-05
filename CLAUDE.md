# YOLOatr -- Deep Learning Based Automatic Target Detection and Localization in Thermal Infrared Imagery

> Paper: arxiv 2507.11267
> Authors: Aon Safdar, Usman Akram, Waseem Anwar, Basit Malik, Mian Ibad Ali
> Venue: 25th Irish Machine Vision and Image Processing Conference (IMVIP2023)
> Module: `anima_yoloatr`

## Paper Summary

YOLOatr is a modified anchor-based single-stage detector for Automatic Target Detection
and Recognition (ATR) in Mid-Wave Infrared (MWIR) thermal imagery. Built on YOLOv5s,
it introduces three key modifications:

1. **Extra small-object detection head (P2)** -- adds a 4th detection scale for tiny IR targets
2. **BiFPN neck** -- replaces PANet with Weighted Bi-directional Feature Pyramid Network
   for better multi-scale feature fusion
3. **Custom Augmentation Profile (CAP)** -- domain-specific augmentation tuned for
   thermal IR imagery (low mosaic, no shear, high mixup/copy-paste)

The model achieves 99.6% mAP@0.5 on correlated DSIAC MWIR and 37.7% on decorrelated
ranges (11.4% gain over baseline YOLOv5s).

## Architecture

### Base Model: YOLOv5s
- Backbone: CSPDarknet53 (small variant)
- Input: 640x640 grayscale (thermal IR)

### YOLOatr Modifications
1. **Neck**: PANet replaced with BiFPN (Weighted Bi-directional FPN)
   - Better feature fusion across scales
   - Learned weighted feature combination
2. **Head**: Added extra small-object detection head (P2)
   - P2 stride=4 (160x160 feature map at 640 input)
   - P3 stride=8, P4 stride=16, P5 stride=32 (original 3 heads)
   - Total: 4 detection heads (P2, P3, P4, P5)
3. **Anchors**: 4 sets of 3 anchors each (12 total, auto-computed via k-means)

### Model Stats
- Parameters: 7,086,449 (~7.1M)
- GFLOPs: 16.4
- Inference: 4.5ms (~110 fps on NVIDIA PC100)

## Dataset: DSIAC MWIR

US Army NVESD ATR Algorithm Development Image Database.

| Property | Value |
|----------|-------|
| Camera | MWIR "cegr" |
| Wavelength | 3-5 um (mid-wave infrared) |
| Total size | 207 GB |
| Targets | 10 vehicles (2 civilian, 8 tactical) + 2 human classes |
| Vehicles used | T72 Tank, BTR70, SUV, Pickup (4 classes) |
| Aspect angles | 72 per vehicle (circle of 100m @ 10mph) |
| Distances | 1000m to 5000m, increments of 500m |
| Time of day | Day and Night |
| Files | 186 ARF format files, 1 min videos @ 30fps (1800 frames/video) |
| Samples per vehicle | 1800 frames x 18 videos = 32,400 frames |
| Total dataset | ~367,200 frames approx |

### Data Split (used in paper)
- Train: 70%
- Validation: 20%
- Test: 10%
- 3,600 images per vehicle type

### Training/Testing Protocols

**DS1 (used in paper)**:
- Training range: 1.0, 1.5, 2.0, 2.5 km (Day+Night) = 10,080 images (1 target), 9,064 (4 targets)
- Correlated test (T1): same ranges = 2,880 (1 target), 2,304 (4 targets)
- Decorrelated test (T2): 3.0 km = 3,600 (1 target), 2,880 (4 targets)

**DS2**:
- Training range: 2.0, 3.5, 4.0, 4.5 km
- Decorrelated test: 5.0 km

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD |
| Learning rate | 0.01 (YOLOv5 default) |
| Momentum | 0.937 (YOLOv5 default) |
| Weight decay | 0.0005 (YOLOv5 default) |
| Batch size | 32 |
| Epochs | 100 |
| Image size | 640x640 |
| Training method | From scratch (random init, 100 epochs) |
| Hardware (paper) | Google Colab Pro, Tesla P100 16GB, 32GB RAM |

### Custom Augmentation Profile (CAP)

| Augmentation | Value |
|-------------|-------|
| fl_gamma (Focal Loss Gamma) | 0.3 |
| hsv_h (Hue) | 0.015 |
| hsv_s (Saturation) | 0.7 |
| hsv_v (Value) | 0.4 |
| degrees (Rotation) | 3 |
| translate | 0.1 |
| scale | 0.3 |
| shear | 0.0 (OFF) |
| perspective | 0.0005 |
| flipud | 0.1 |
| fliplr | 0.5 |
| mosaic | 0.1 (LOW) |
| mixup | 0.4 |
| copy_paste | 0.5 |

## Loss Functions

YOLOv5 standard losses (anchor-based):
1. **Box loss**: CIoU (Complete Intersection over Union)
2. **Objectness loss**: BCE (Binary Cross-Entropy) with focal loss (gamma=0.3)
3. **Classification loss**: BCE (Binary Cross-Entropy)

Loss weights follow YOLOv5s defaults:
- box: 0.05
- obj: 1.0 (scaled per detection layer)
- cls: 0.5

## Evaluation Metrics

- **Precision** = TP / All Detections
- **Recall** = TP / All Ground Truths
- **mAP@0.5** = mean Average Precision at IoU threshold 0.5

### Results (DS1, 4 targets)

**Correlated (T1) -- same range testing**:
| Target | Precision | Recall | mAP@0.5 |
|--------|-----------|--------|---------|
| T72 Tank | 0.996 | 0.995 | 0.997 |
| BTR70 | 0.999 | 1.00 | 0.996 |
| SUV | 0.996 | 0.993 | 0.994 |
| Pickup | 1.00 | 0.999 | 0.995 |
| **All** | **0.996** | **0.997** | **0.996** |

**Decorrelated (T2) -- unseen range testing**:
| Target | Precision | Recall | mAP@0.5 |
|--------|-----------|--------|---------|
| T72 Tank | 0.699 | 0.618 | 0.622 |
| BTR70 | 0.169 | 0.132 | 0.214 |
| SUV | 0.562 | 0.373 | 0.393 |
| Pickup | 0.278 | 0.435 | 0.163 |
| **All** | **0.512** | **0.44** | **0.377** |

### vs Baseline YOLOv5s
- Correlated: +0.02% mAP gain (99.4% -> 99.6%)
- Decorrelated: +11.4% mAP gain (26.3% -> 37.7%)

## Model Requirements

- Base: YOLOv5s (ultralytics) -- modified
- No pretrained weights needed (trained from scratch)
- PyTorch >= 1.7

## Defense Module Context

This module addresses thermal IR target detection for defense/surveillance:
- Real-time ATR on tactical platforms with limited compute
- Robust to range, weather, illumination, viewpoint variations
- Handles small targets at long range (1-5 km)
- MWIR modality (3-5 um wavelength)
