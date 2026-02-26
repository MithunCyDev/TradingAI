#!/usr/bin/env python3
"""
Run the HQTS FastAPI server.

Uses uvicorn. Configure API_HOST and API_PORT via .env.
"""

import os
import sys
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
