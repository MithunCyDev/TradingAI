#!/usr/bin/env python3
"""Run HQTS feature and labeling pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hqts.features.pipeline import main

if __name__ == "__main__":
    main()
