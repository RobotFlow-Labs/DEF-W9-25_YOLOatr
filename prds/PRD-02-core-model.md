# PRD-02: Core Model

> Status: TODO
> Module: anima_yoloatr

## Objective
Implement the full YOLOatr architecture: CSPDarknet53-Small backbone, BiFPN neck,
4-head detection (P2/P3/P4/P5), and anchor generation.

## Architecture Details

### Backbone: CSPDarknet53-Small
- Focus layer (stem): 12x3 -> 32 channels, stride 2
- CBS blocks (Conv-BN-SiLU)
- C3 blocks (CSP Bottleneck with 3 convolutions)
- Stages: [64, 128, 256, 512] channels
- SPPF (Spatial Pyramid Pooling - Fast) at the end

### Neck: BiFPN (Weighted Bi-directional FPN)
- Replaces original PANet in YOLOv5s
- Weighted feature fusion with learnable weights (fast normalized fusion)
- Top-down + bottom-up pathways
- Outputs at P2 (stride 4), P3 (stride 8), P4 (stride 16), P5 (stride 32)

### Detection Heads (4 scales)
- P2: 160x160 feature map, 3 anchors -- extra small object head
- P3: 80x80 feature map, 3 anchors
- P4: 40x40 feature map, 3 anchors
- P5: 20x20 feature map, 3 anchors
- Each head outputs: [batch, num_anchors, grid_h, grid_w, 5 + num_classes]

### Anchors
- Auto-computed via k-means on training data
- 4 sets of 3 anchors (12 total)
- Default anchors (YOLOv5s-style, to be refined):
  - P2: small anchors for tiny targets
  - P3: [10,13, 16,30, 33,23]
  - P4: [30,61, 62,45, 59,119]
  - P5: [116,90, 156,198, 373,326]

### Model Stats
- Parameters: ~7.1M
- GFLOPs: 16.4

## Deliverables
- [ ] src/anima_yoloatr/model.py -- full YOLOatr model
- [ ] src/anima_yoloatr/blocks.py -- CBS, C3, SPPF, BiFPN blocks
- [ ] Anchor auto-computation utility
- [ ] Model summary matching paper stats (7.1M params, 16.4 GFLOPs)

## Acceptance Criteria
- Forward pass with random input (1, 3, 640, 640) produces valid output
- Parameter count within 5% of paper (7,086,449)
- GFLOPs within 10% of paper (16.4)
- All 4 detection heads produce correct output shapes
