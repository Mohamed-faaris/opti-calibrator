#!/usr/bin/env python3
"""
CLI launcher for syncing camera device nodes.
Usage:
    python sync_devices.py [--dry-run] [--output-dir output]
"""

import argparse
import sys
from pathlib import Path

# Ensure package directory is on path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from calibrator.sync_devices import sync_device_nodes, print_sync_table


def main():
    parser = argparse.ArgumentParser(
        description="Sync camera device nodes in calibration profiles by matching card names and streams."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Path to output calibrations directory (default: 'output')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying calibration files"
    )

    args = parser.parse_args()
    results = sync_device_nodes(output_dir=Path(args.output_dir), dry_run=args.dry_run)
    print_sync_table(results)


if __name__ == "__main__":
    main()
