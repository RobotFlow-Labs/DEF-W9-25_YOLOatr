# PRD-03: Loss Functions

> Status: TODO
> Module: anima_yoloatr

## Objective
Implement YOLOv5-style losses adapted for YOLOatr's 4-head architecture.

## Loss Components

### 1. Box Regression Loss: CIoU
- Complete IoU loss for bounding box regression
- CIoU = IoU - (rho^2(b, b_gt) / c^2) - alpha * v
  - rho: Euclidean distance between box centers
  - c: diagonal of smallest enclosing box
  - v: aspect ratio consistency term
  - alpha: trade-off parameter

### 2. Objectness Loss: BCE with Focal Loss
- Binary Cross-Entropy for objectness prediction
- Focal Loss modifier with gamma = 0.3 (from CAP)
- FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
- Objectness scaled per detection layer:
  - Larger stride layers get higher objectness weight
  - Default: P2=4.0, P3=1.0, P4=0.4, P5=0.1 (approximate)

### 3. Classification Loss: BCE
- Binary Cross-Entropy for multi-class classification
- Independent sigmoid per class (not softmax)

### Loss Weights (YOLOv5s defaults)
- box_loss_gain: 0.05
- obj_loss_gain: 1.0
- cls_loss_gain: 0.5

### Build Targets
- Anchor matching: IoU-based matching between GT and anchors
- Positive samples: GT matched to anchors with IoU > threshold
- Grid assignment: GT center determines responsible grid cell

## Deliverables
- [ ] src/anima_yoloatr/losses.py -- CIoU, Focal BCE, ComputeLoss class
- [ ] Build targets function for 4-head anchor matching
- [ ] Loss weight configuration from TOML

## Acceptance Criteria
- CIoU loss produces correct gradients
- Focal loss with gamma=0.3 matches expected behavior
- Loss computation handles all 4 detection scales
- Total loss is scalar, backprop works
