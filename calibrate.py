#!/usr/bin/env python3
"""
Camera Calibrator launcher.
"""

import sys
from pathlib import Path

# Ensure package directory is on path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from calibrator.cli import main

if __name__ == "__main__":
    main()
