#!/usr/bin/env python3
"""Run HQTS ETL extraction pipeline."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hqts.etl.extract import main

if __name__ == "__main__":
    main()
