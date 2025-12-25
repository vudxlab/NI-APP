#!/usr/bin/env python3
"""
Launcher script for NI DAQ Vibration Analysis application.

This script properly sets up the Python path and launches the application.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the main application
from src.main import main

if __name__ == "__main__":
    main()

