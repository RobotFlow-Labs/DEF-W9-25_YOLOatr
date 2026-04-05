# PRD-05: Evaluation

> Status: TODO
> Module: anima_yoloatr

## Objective
Implement evaluation pipeline with paper metrics: precision, recall, mAP@0.5,
per-class metrics, and support for correlated/decorrelated test protocols.

## Metrics
1. **Precision** = TP / (TP + FP)
2. **Recall** = TP / (TP + FN)
3. **AP per class** = area under precision-recall curve
4. **mAP@0.5** = mean of AP across all classes at IoU=0.5
5. **F1** = 2 * P * R / (P + R)

## Test Protocols
- **Correlated (T1)**: test on same ranges as training (1.0-2.5 km)
- **Decorrelated (T2)**: test on unseen range (3.0 km or 5.0 km)

## Paper Targets (to reproduce)
| Protocol | Precision | Recall | mAP@0.5 |
|----------|-----------|--------|---------|
| Correlated (T1) | 0.996 | 0.997 | 0.996 |
| Decorrelated (T2) | 0.512 | 0.44 | 0.377 |

## Deliverables
- [ ] src/anima_yoloatr/evaluate.py -- evaluation engine
- [ ] scripts/evaluate.py -- CLI entry point
- [ ] Per-class precision, recall, AP computation
- [ ] Confusion matrix generation
- [ ] PR curve plotting
- [ ] Support for correlated and decorrelated protocols
- [ ] Report generation (markdown)

## Acceptance Criteria
- mAP computation matches standard COCO-style evaluation
- Per-class metrics reported correctly
- Confusion matrix visualized
- Report saved to /mnt/artifacts-datai/reports/project_yoloatr/
