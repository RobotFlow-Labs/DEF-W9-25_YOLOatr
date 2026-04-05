#!/usr/bin/env python3
"""YOLOatr evaluation entry point.

Usage:
    # Evaluate on test set (correlated protocol)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/evaluate.py \
        --config configs/paper.toml \
        --weights /mnt/artifacts-datai/checkpoints/project_yoloatr/best.pth \
        --split test --protocol correlated

    # Evaluate on decorrelated protocol
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/evaluate.py \
        --config configs/paper.toml \
        --weights /mnt/artifacts-datai/checkpoints/project_yoloatr/best.pth \
        --split test --protocol decorrelated
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anima_yoloatr.evaluate import main

if __name__ == "__main__":
    main()
