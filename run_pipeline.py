#!/usr/bin/env python
"""
================================================================================
PPP-Q INCREMENTAL ML PIPELINE - RUN SCRIPT
================================================================================
Entry point to run the INCREMENTAL automated ML pipeline

The pipeline will:
  1. Check existing data in final_consolidated_dataset.csv
  2. Fetch ONLY new data since the last date
  3. Append new rows to the consolidated dataset
  4. Preprocess only the new rows
  5. Retrain models on ALL data (existing + new)
  6. Send email notification to ba8616127@gmail.com

Usage:
    python run_pipeline.py              # Run incremental update
    python run_pipeline.py --force      # Force full retrain even if no new data
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
    parser = argparse.ArgumentParser(description='PPP-Q Incremental ML Pipeline')
    parser.add_argument('--schedule', action='store_true', 
                       help='Run on schedule (every 15 days)')
    parser.add_argument('--test-email', action='store_true',
                       help='Send test email to verify configuration')
    parser.add_argument('--force', action='store_true',
                       help='Force full retrain even if no new data available')
    
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
        mode = "FORCED FULL" if args.force else "INCREMENTAL"
        print(f"🚀 Running {mode} pipeline...")
        print("   → Will fetch ONLY new data since last update")
        print("   → Will append new rows to existing dataset")
        print("   → Will preprocess only new rows")
        print("   → Will retrain on all data (existing + new)")
        print()
        
        from src.pipelines.prefect_flows import run_pipeline
        result = run_pipeline(force_full_retrain=args.force)
        
        if result.get('status') == 'skipped':
            print(f"\n⚠️ Pipeline skipped: {result.get('reason', 'no new data')}")
            print("   Use --force to run anyway")
        else:
            print(f"\n✅ Pipeline completed!")
            print(f"   Best Model: {result.get('best_model', 'N/A')}")
            print(f"   Macro F1: {result.get('metrics', {}).get('macro_f1', 'N/A')}")
            print(f"   Deployed: {result.get('deployed', False)}")


if __name__ == "__main__":
    main()
