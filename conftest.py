"""
Root pytest configuration - sets up Python path before any imports.
"""

import sys
from pathlib import Path

# Add src to sys.path BEFORE pytest collects any tests
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
