#!/usr/bin/env python3
"""Deployment script for PPPQ ML project"""

import subprocess
import sys
from pathlib import Path

def run_cmd(cmd):
    """Run command and return success status"""
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=Path.cwd())
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🚀 PPPQ Deployment")
    
    # Check model files
    if not Path("models/pppq/lgbm_model.txt").exists():
        print("⚠️  Model files not found")
        return False
    
    # Check data files
    if not Path("data/processed/pppq/train/pppq_train.csv").exists():
        print("⚠️  Data files not found")
        return False
    
    print("✓ All checks passed")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
