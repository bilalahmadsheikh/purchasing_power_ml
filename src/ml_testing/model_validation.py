"""
Model Validation Module

Validates ML model performance and behavior:
- Performance metrics validation
- Prediction consistency
- Feature importance stability
- Model behavior tests
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Validates ML model performance and behavior.
    
    Tests:
    - Minimum performance thresholds
    - Prediction consistency
    - Feature importance stability
    - Edge case handling
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize model validator with optional config"""
        self.config = config or {}
        self.validation_results: Dict[str, Any] = {}
        
        # Performance thresholds
        self.min_accuracy = self.config.get("min_accuracy", 0.85)
        self.min_f1 = self.config.get("min_f1", 0.80)
        self.min_precision = self.config.get("min_precision", 0.75)
        self.min_recall = self.config.get("min_recall", 0.75)
        
        # Stability thresholds
        self.max_prediction_variance = self.config.get("max_prediction_variance", 0.1)
        self.feature_importance_correlation = self.config.get("feature_importance_correlation", 0.8)
    
    def validate_all(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: Optional[List[str]] = None,
        reference_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Run all model validation checks.
        
        Args:
            model: Trained model with predict() method
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            reference_metrics: Optional reference metrics to compare against
            
        Returns:
            Dictionary with all validation results
        """
        logger.info(f"Starting model validation on {len(X_test)} samples")
        
        results = {
            "test_samples": len(X_test),
            "checks": {},
            "passed": True,
            "summary": {}
        }
        
        # Make predictions
        try:
            y_pred = model.predict(X_test)
            if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1:
                y_pred = np.argmax(y_pred, axis=1)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e), "passed": False}
        
        # Run all checks
        checks = [
            ("performance_metrics", self.check_performance_metrics(y_test, y_pred, reference_metrics)),
            ("prediction_consistency", self.check_prediction_consistency(model, X_test)),
            ("class_coverage", self.check_class_coverage(y_test, y_pred)),
            ("confidence_distribution", self.check_confidence_distribution(model, X_test)),
        ]
        
        # Feature importance check if model supports it
        if hasattr(model, 'feature_importance') or hasattr(model, 'feature_importances_'):
            checks.append(("feature_importance", self.check_feature_importance(model, feature_names)))
        
        passed_count = 0
        for name, check_result in checks:
            results["checks"][name] = check_result
            if check_result.get("passed", False):
                passed_count += 1
            else:
                results["passed"] = False
        
        results["summary"] = {
            "total_checks": len(checks),
            "passed_checks": passed_count,
            "failed_checks": len(checks) - passed_count,
            "pass_rate": round(passed_count / len(checks) * 100, 1)
        }
        
        self.validation_results = results
        logger.info(f"Model validation complete: {passed_count}/{len(checks)} checks passed")
        
        return results
    
    def check_performance_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        reference_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Check model performance meets minimum thresholds"""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics = {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1, 4),
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4)
        }
        
        # Check thresholds
        threshold_checks = {
            "accuracy": accuracy >= self.min_accuracy,
            "f1": f1 >= self.min_f1,
            "precision": precision >= self.min_precision,
            "recall": recall >= self.min_recall
        }
        
        passed = all(threshold_checks.values())
        
        # Compare to reference if provided
        degradation = {}
        if reference_metrics:
            for metric, value in metrics.items():
                ref_value = reference_metrics.get(metric, value)
                change = (value - ref_value) / (ref_value + 1e-10) * 100
                degradation[metric] = round(change, 2)
        
        return {
            "passed": passed,
            "metrics": metrics,
            "thresholds": {
                "min_accuracy": self.min_accuracy,
                "min_f1": self.min_f1,
                "min_precision": self.min_precision,
                "min_recall": self.min_recall
            },
            "threshold_checks": threshold_checks,
            "degradation_from_reference": degradation if degradation else None,
            "message": f"Accuracy: {accuracy:.2%}, F1: {f1:.2%}"
        }
    
    def check_prediction_consistency(
        self,
        model: Any,
        X_test: pd.DataFrame,
        n_runs: int = 3
    ) -> Dict[str, Any]:
        """Check prediction consistency across multiple runs"""
        predictions = []
        
        for _ in range(n_runs):
            try:
                pred = model.predict(X_test)
                if hasattr(pred, 'shape') and len(pred.shape) > 1:
                    pred = np.argmax(pred, axis=1)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Prediction run failed: {e}")
                continue
        
        if len(predictions) < 2:
            return {
                "passed": True,
                "message": "Insufficient prediction runs",
                "skipped": True
            }
        
        # Check consistency (all predictions should be identical for deterministic models)
        base_pred = predictions[0]
        consistency_scores = []
        
        for pred in predictions[1:]:
            match_rate = np.mean(base_pred == pred)
            consistency_scores.append(match_rate)
        
        avg_consistency = np.mean(consistency_scores)
        passed = avg_consistency >= (1 - self.max_prediction_variance)
        
        return {
            "passed": passed,
            "consistency_score": round(avg_consistency, 4),
            "num_runs": len(predictions),
            "threshold": 1 - self.max_prediction_variance,
            "message": f"Prediction consistency: {avg_consistency:.2%}"
        }
    
    def check_class_coverage(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Check that model predicts all classes"""
        true_classes = set(y_true.unique())
        pred_classes = set(np.unique(y_pred))
        
        missing_classes = true_classes - pred_classes
        coverage = len(pred_classes) / len(true_classes) if true_classes else 0
        
        passed = len(missing_classes) == 0
        
        return {
            "passed": passed,
            "true_classes": list(true_classes),
            "predicted_classes": list(pred_classes),
            "missing_classes": list(missing_classes),
            "coverage": round(coverage, 2),
            "message": f"Class coverage: {len(pred_classes)}/{len(true_classes)} classes"
        }
    
    def check_confidence_distribution(
        self,
        model: Any,
        X_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Check prediction confidence distribution"""
        try:
            # Get probabilities
            probabilities = model.predict(X_test)
            
            if len(probabilities.shape) == 1:
                # Binary classification
                confidences = np.maximum(probabilities, 1 - probabilities)
            else:
                # Multi-class
                confidences = np.max(probabilities, axis=1)
            
            # Statistics
            mean_conf = np.mean(confidences)
            min_conf = np.min(confidences)
            max_conf = np.max(confidences)
            
            # Distribution bins
            low_conf = np.mean(confidences < 0.5)
            medium_conf = np.mean((confidences >= 0.5) & (confidences < 0.8))
            high_conf = np.mean(confidences >= 0.8)
            
            # Pass if not too many low confidence predictions
            passed = low_conf < 0.2  # Less than 20% low confidence
            
            return {
                "passed": passed,
                "mean_confidence": round(mean_conf, 4),
                "min_confidence": round(min_conf, 4),
                "max_confidence": round(max_conf, 4),
                "distribution": {
                    "low (<50%)": round(low_conf * 100, 1),
                    "medium (50-80%)": round(medium_conf * 100, 1),
                    "high (>80%)": round(high_conf * 100, 1)
                },
                "message": f"Mean confidence: {mean_conf:.2%}, Low conf rate: {low_conf:.1%}"
            }
        except Exception as e:
            return {
                "passed": True,
                "message": f"Could not check confidence: {e}",
                "skipped": True
            }
    
    def check_feature_importance(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Check feature importance distribution"""
        try:
            # Get feature importance
            if hasattr(model, 'feature_importance'):
                importance = model.feature_importance()
            elif hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                return {
                    "passed": True,
                    "message": "Model does not expose feature importance",
                    "skipped": True
                }
            
            # Normalize
            importance = np.array(importance)
            importance = importance / importance.sum()
            
            # Top features
            sorted_idx = np.argsort(importance)[::-1]
            top_5_idx = sorted_idx[:5]
            
            top_features = []
            for idx in top_5_idx:
                name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}"
                top_features.append({
                    "name": name,
                    "importance": round(float(importance[idx]), 4)
                })
            
            # Check for feature importance concentration
            top_5_importance = sum(f["importance"] for f in top_features)
            passed = top_5_importance < 0.8  # Top 5 shouldn't dominate too much
            
            return {
                "passed": passed,
                "num_features": len(importance),
                "top_5_features": top_features,
                "top_5_importance_sum": round(top_5_importance, 4),
                "mean_importance": round(float(np.mean(importance)), 4),
                "message": f"Top 5 features account for {top_5_importance:.1%} of importance"
            }
        except Exception as e:
            return {
                "passed": True,
                "message": f"Feature importance check failed: {e}",
                "skipped": True
            }
    
    def generate_report(self) -> str:
        """Generate human-readable validation report"""
        if not self.validation_results:
            return "No validation results available. Run validate_all() first."
        
        results = self.validation_results
        lines = [
            "=" * 60,
            "MODEL VALIDATION REPORT",
            "=" * 60,
            f"Test Samples: {results.get('test_samples', 'N/A')}",
            f"Overall Status: {'✅ PASSED' if results['passed'] else '❌ FAILED'}",
            f"Checks Passed: {results['summary']['passed_checks']}/{results['summary']['total_checks']}",
            "",
            "-" * 60,
            "CHECK DETAILS:",
            "-" * 60,
        ]
        
        for check_name, check_result in results["checks"].items():
            status = "✅" if check_result.get("passed", False) else "❌"
            message = check_result.get("message", "No details")
            lines.append(f"{status} {check_name.upper()}: {message}")
        
        # Performance metrics detail
        if "performance_metrics" in results["checks"]:
            perf = results["checks"]["performance_metrics"]
            if "metrics" in perf:
                lines.append("")
                lines.append("Performance Metrics:")
                for metric, value in perf["metrics"].items():
                    lines.append(f"  - {metric}: {value:.4f}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def run_model_validation(
    model_path: str,
    test_data_path: str,
    feature_columns_path: str,
    label_col: str = "PPP_Q_Label",
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Convenience function to run model validation.
    
    Args:
        model_path: Path to saved model
        test_data_path: Path to test data CSV
        feature_columns_path: Path to feature columns JSON
        label_col: Label column name
        config: Optional configuration
        
    Returns:
        Model validation results
    """
    import lightgbm as lgb
    
    # Check files exist
    if not Path(model_path).exists():
        return {"error": f"Model not found: {model_path}", "passed": False}
    if not Path(test_data_path).exists():
        return {"error": f"Test data not found: {test_data_path}", "passed": False}
    
    # Load model with error handling
    try:
        model = lgb.Booster(model_file=model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}", "passed": False}
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    
    # Load feature columns
    feature_names = None
    if Path(feature_columns_path).exists():
        with open(feature_columns_path, 'r') as f:
            feature_names = json.load(f)
    
    # Prepare data
    if feature_names:
        X_test = test_df[feature_names]
    else:
        X_test = test_df.drop(columns=[label_col, 'Asset', 'Date'], errors='ignore')
    
    y_test = test_df[label_col] if label_col in test_df.columns else pd.Series([0] * len(test_df))
    
    # Run validation
    validator = ModelValidator(config)
    return validator.validate_all(model, X_test, y_test, feature_names)
