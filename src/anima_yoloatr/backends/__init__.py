"""YOLOatr backend auto-detection.

Selects CUDA > MLX > CPU backend automatically.
"""

from __future__ import annotations

import os


def get_backend() -> str:
    """Detect and return the best available backend.

    Returns:
        One of: "cuda", "mlx", "cpu"
    """
    forced = os.environ.get("ANIMA_BACKEND", "auto")
    if forced != "auto":
        return forced

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    try:
        import mlx.core  # noqa: F401

        return "mlx"
    except ImportError:
        pass

    return "cpu"
