#!/usr/bin/env python
"""Export YOLOatr model to all formats.

Usage:
    python scripts/export.py --checkpoint /path/to/best.pth
    python scripts/export.py --checkpoint /path/to/best.pth --config configs/nuaa_sirst.toml
"""

from anima_yoloatr.export import main

if __name__ == "__main__":
    main()
