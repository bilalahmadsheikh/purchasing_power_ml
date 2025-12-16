"""
================================================================================
AUTOMATED PIPELINE SYSTEM
================================================================================
Complete automated data pipeline for PPP-Q model

Components:
- pipeline_config.py: Central configuration
- notifications.py: Email notifications
- model_registry.py: MLflow model versioning
- prefect_flows.py: Main Prefect orchestration

Schedule: Runs automatically every 15 days

Author: Bilal Ahmad Sheikh
Date: December 2024
================================================================================
"""

from .pipeline_config import PipelineConfig
from .notifications import NotificationManager, notifier
from .model_registry import ModelRegistry
from .prefect_flows import pppq_ml_pipeline, run_pipeline

__all__ = [
    'PipelineConfig',
    'NotificationManager',
    'notifier',
    'ModelRegistry',
    'pppq_ml_pipeline',
    'run_pipeline'
]
