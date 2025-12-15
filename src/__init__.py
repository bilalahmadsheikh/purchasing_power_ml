"""
PPP-Q Investment Classifier - Production MLOps System
Author: Bilal Ahmad Sheikh
"""

__version__ = "1.0.0"
__author__ = "Bilal Ahmad Sheikh"
__email__ = "bilal@example.com"

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Key directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Create directories if they don't exist
for directory in [LOGS_DIR, MODELS_DIR / "pppq"]:
    directory.mkdir(parents=True, exist_ok=True)