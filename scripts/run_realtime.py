#!/usr/bin/env python
"""Convenience launcher for the real-time demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.realtime import run_demo

if __name__ == "__main__":
    run_demo()
