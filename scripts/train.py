#!/usr/bin/env python3
"""YOLOatr training entry point.

Usage:
    # Paper config (full training)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py --config configs/paper.toml

    # Debug / smoke test
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py --config configs/debug.toml

    # Resume from checkpoint
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py --config configs/paper.toml \
        --resume /mnt/artifacts-datai/checkpoints/project_yoloatr/best.pth
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anima_yoloatr.train import main

if __name__ == "__main__":
    main()
