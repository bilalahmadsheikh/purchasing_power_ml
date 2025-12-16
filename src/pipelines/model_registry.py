"""
================================================================================
MODEL REGISTRY
================================================================================
MLflow-based model versioning and tracking

Author: Bilal Ahmad Sheikh
Date: December 2024
================================================================================
"""

import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Model registry for version tracking and management"""
    
    def __init__(self):
        self.config = PipelineConfig
        self.registry_path = self.config.MODEL_DIR / "model_registry.json"
        self.registry = self._load_registry()
        
        # Try to initialize MLflow
        self.mlflow_available = False
        try:
            import mlflow
            self.mlflow = mlflow
            mlflow.set_tracking_uri(str(self.config.MLFLOW_TRACKING_URI))
            
            # Create or get experiment
            existing_exp = mlflow.get_experiment_by_name(self.config.MLFLOW_EXPERIMENT_NAME)
            if existing_exp:
                self.experiment_id = existing_exp.experiment_id
            else:
                self.experiment_id = mlflow.create_experiment(self.config.MLFLOW_EXPERIMENT_NAME)
            
            self.mlflow_available = True
            logger.info(f"✅ MLflow initialized: Experiment '{self.config.MLFLOW_EXPERIMENT_NAME}'")
        except ImportError:
            logger.warning("⚠️ MLflow not installed, using local registry only")
        except Exception as e:
            logger.warning(f"⚠️ MLflow initialization failed: {e}, using local registry only")
    
    def _load_registry(self) -> Dict:
        """Load registry from disk"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load registry: {e}")
        
        return {
            'models': [],
            'current_production': None,
            'current_staging': None
        }
    
    def _save_registry(self):
        """Save registry to disk"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def log_model_training(
        self,
        model_name: str,
        model_type: str,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        params: Dict[str, Any],
        feature_importance: Optional[Dict] = None
    ) -> str:
        """
        Log model training run
        
        Args:
            model_name: Name of the model
            model_type: Type (lightgbm, xgboost, etc.)
            train_metrics: Training metrics
            val_metrics: Validation metrics
            test_metrics: Test metrics
            params: Model parameters
            feature_importance: Feature importance dict
        
        Returns:
            Run ID
        """
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_entry = {
            'run_id': run_id,
            'model_name': model_name,
            'model_type': model_type,
            'training_date': datetime.now().isoformat(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'params': {k: str(v) for k, v in params.items()},
            'stage': 'None'
        }
        
        self.registry['models'].append(model_entry)
        self._save_registry()
        
        # Log to MLflow if available
        if self.mlflow_available:
            try:
                with self.mlflow.start_run(experiment_id=self.experiment_id, run_name=f"{model_name}_{run_id}"):
                    # Log parameters
                    for param_name, param_value in params.items():
                        if isinstance(param_value, (int, float, str, bool)):
                            self.mlflow.log_param(param_name, param_value)
                    
                    # Log metrics
                    for metric_name, metric_value in train_metrics.items():
                        self.mlflow.log_metric(f"train_{metric_name}", metric_value)
                    for metric_name, metric_value in val_metrics.items():
                        self.mlflow.log_metric(f"val_{metric_name}", metric_value)
                    for metric_name, metric_value in test_metrics.items():
                        self.mlflow.log_metric(f"test_{metric_name}", metric_value)
                    
                    # Log feature importance as artifact
                    if feature_importance:
                        importance_path = self.config.REPORTS_DIR / 'feature_importance.json'
                        with open(importance_path, 'w') as f:
                            json.dump(feature_importance, f)
                        self.mlflow.log_artifact(str(importance_path))
                    
                    logger.info(f"✅ Model logged to MLflow: {run_id}")
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
        
        logger.info(f"✅ Model training logged: {run_id}")
        return run_id
    
    def get_latest_model(self, stage: str = 'Production') -> Optional[Dict]:
        """Get latest model in specified stage"""
        if stage == 'Production':
            run_id = self.registry.get('current_production')
        elif stage == 'Staging':
            run_id = self.registry.get('current_staging')
        else:
            run_id = None
        
        if run_id:
            for model in self.registry['models']:
                if model['run_id'] == run_id:
                    return model
        
        return None
    
    def promote_to_staging(self, run_id: str) -> bool:
        """Promote model to staging"""
        for model in self.registry['models']:
            if model['run_id'] == run_id:
                # Demote current staging
                if self.registry['current_staging']:
                    for m in self.registry['models']:
                        if m['run_id'] == self.registry['current_staging']:
                            m['stage'] = 'Archived'
                
                model['stage'] = 'Staging'
                self.registry['current_staging'] = run_id
                self._save_registry()
                logger.info(f"✅ Model {run_id} promoted to Staging")
                return True
        
        logger.error(f"Model {run_id} not found")
        return False
    
    def promote_to_production(self, run_id: str) -> bool:
        """Promote model to production"""
        for model in self.registry['models']:
            if model['run_id'] == run_id:
                # Archive current production
                if self.registry['current_production']:
                    for m in self.registry['models']:
                        if m['run_id'] == self.registry['current_production']:
                            m['stage'] = 'Archived'
                
                model['stage'] = 'Production'
                self.registry['current_production'] = run_id
                self._save_registry()
                logger.info(f"✅ Model {run_id} promoted to Production")
                return True
        
        logger.error(f"Model {run_id} not found")
        return False
    
    def compare_models(self, run_id_1: str, run_id_2: str) -> Dict[str, Any]:
        """Compare metrics between two model runs"""
        model_1 = None
        model_2 = None
        
        for model in self.registry['models']:
            if model['run_id'] == run_id_1:
                model_1 = model
            elif model['run_id'] == run_id_2:
                model_2 = model
        
        if not model_1 or not model_2:
            return {'error': 'One or both models not found'}
        
        comparison = {
            'model_1': {
                'run_id': run_id_1,
                'test_metrics': model_1['test_metrics']
            },
            'model_2': {
                'run_id': run_id_2,
                'test_metrics': model_2['test_metrics']
            },
            'improvements': {}
        }
        
        for key in model_1['test_metrics'].keys():
            if key in model_2['test_metrics']:
                improvement = model_2['test_metrics'][key] - model_1['test_metrics'][key]
                comparison['improvements'][key] = improvement
        
        return comparison
    
    def get_all_models(self) -> list:
        """Get all registered models"""
        return self.registry['models']
    
    def get_model_metrics(self, run_id: str) -> Optional[Dict]:
        """Get metrics for a specific model"""
        for model in self.registry['models']:
            if model['run_id'] == run_id:
                return model['test_metrics']
        return None
    
    def should_deploy(self, new_metrics: Dict, threshold: float = 0.001) -> Tuple[bool, Optional[Dict]]:
        """
        Check if new model should be deployed based on improvement
        
        Args:
            new_metrics: New model's test metrics
            threshold: Minimum F1 improvement required
        
        Returns:
            (should_deploy: bool, previous_metrics: Dict or None)
        """
        current_production = self.get_latest_model('Production')
        
        if not current_production:
            logger.info("No production model exists, recommending deployment")
            return True, None
        
        prev_f1 = current_production['test_metrics'].get('macro_f1', 0)
        new_f1 = new_metrics.get('macro_f1', 0)
        
        improvement = new_f1 - prev_f1
        
        if improvement >= threshold:
            logger.info(f"✅ Improvement detected: {improvement:.4f} (threshold: {threshold})")
            return True, current_production['test_metrics']
        else:
            logger.warning(f"⚠️ Improvement below threshold: {improvement:.4f} < {threshold}")
            return False, current_production['test_metrics']


# Global registry instance
registry = ModelRegistry()
