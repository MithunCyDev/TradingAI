#!/usr/bin/env python3
"""Run HQTS model training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hqts.models.train import main

if __name__ == "__main__":
    main()
