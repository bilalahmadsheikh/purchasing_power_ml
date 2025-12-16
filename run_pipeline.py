#!/usr/bin/env python
"""
================================================================================
PPP-Q ML PIPELINE - RUN SCRIPT
================================================================================
Simple entry point to run the automated ML pipeline

Usage:
    python run_pipeline.py          # Run once
    python run_pipeline.py --schedule   # Run on schedule (every 15 days)
    python run_pipeline.py --test-email # Send test email

Author: Bilal Ahmad Sheikh
Date: December 2024
================================================================================
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description='PPP-Q ML Pipeline')
    parser.add_argument('--schedule', action='store_true', 
                       help='Run on schedule (every 15 days)')
    parser.add_argument('--test-email', action='store_true',
                       help='Send test email to verify configuration')
    
    args = parser.parse_args()
    
    if args.test_email:
        print("📧 Sending test email...")
        from src.pipelines.notifications import notifier
        success = notifier.send_test_email()
        if success:
            print("✅ Test email sent successfully!")
        else:
            print("❌ Failed to send test email. Check your .env configuration.")
        return
    
    if args.schedule:
        print("📅 Starting scheduled pipeline (runs every 15 days)...")
        from src.pipelines.prefect_flows import schedule_pipeline
        schedule_pipeline()
    else:
        print("🚀 Running pipeline once...")
        from src.pipelines.prefect_flows import run_pipeline
        result = run_pipeline()
        print(f"\n✅ Pipeline completed!")
        print(f"   Best Model: {result.get('best_model', 'N/A')}")
        print(f"   Macro F1: {result.get('metrics', {}).get('macro_f1', 'N/A')}")
        print(f"   Deployed: {result.get('deployed', False)}")


if __name__ == "__main__":
    main()
