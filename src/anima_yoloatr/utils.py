"""YOLOatr utility functions."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_config(config_path: str) -> dict[str, Any]:
    """Load TOML configuration file.

    Args:
        config_path: Path to .toml config file

    Returns:
        Configuration dictionary
    """
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "rb") as f:
        return tomllib.load(f)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        boxes: [N, 4] in xywh format

    Returns:
        [N, 4] in xyxy format
    """
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        boxes: [N, 4] in xyxy format

    Returns:
        [N, 4] in xywh format
    """
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return torch.stack([cx, cy, w, h], dim=1)


def compute_model_stats(model: torch.nn.Module, input_size: int = 640) -> dict:
    """Compute model parameter count and approximate FLOPs.

    Args:
        model: PyTorch model
        input_size: Input image size

    Returns:
        Dict with param_count, param_count_trainable
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_m": total_params / 1e6,
    }


def check_gpu_memory(max_util: float = 0.80) -> None:
    """Check GPU memory utilization.

    Raises RuntimeError if any GPU exceeds max_util.

    Args:
        max_util: Maximum VRAM utilization fraction (default 0.80)
    """
    if not torch.cuda.is_available():
        return

    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_mem
        used = torch.cuda.memory_allocated(i)
        util = used / total
        if util > max_util:
            raise RuntimeError(
                f"GPU {i} at {util * 100:.1f}% VRAM -- exceeds {max_util * 100:.0f}% cap. "
                f"Reduce batch_size or enable gradient checkpointing."
            )


def make_output_dirs(config: dict) -> None:
    """Create all output directories specified in config.

    Args:
        config: Configuration dictionary
    """
    dirs = [
        config.get("checkpoint", {}).get("output_dir", ""),
        config.get("logging", {}).get("log_dir", ""),
        config.get("logging", {}).get("tensorboard_dir", ""),
        config.get("evaluation", {}).get("report_dir", ""),
        config.get("export", {}).get("output_dir", ""),
    ]
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)
