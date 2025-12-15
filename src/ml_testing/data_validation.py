"""
Data Validation Module

Uses DeepChecks for comprehensive data integrity testing:
- Missing values detection
- Duplicate detection
- Feature type validation
- Outlier detection
- Data distribution checks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates data integrity for ML pipeline.
    
    Checks:
    - Missing values and patterns
    - Duplicate rows
    - Feature type consistency
    - Value ranges and outliers
    - Data distribution health
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize validator with optional config"""
        self.config = config or {}
        self.validation_results: Dict[str, Any] = {}
        
        # Default thresholds
        self.max_missing_pct = self.config.get("max_missing_pct", 5.0)
        self.max_duplicate_pct = self.config.get("max_duplicate_pct", 1.0)
        self.outlier_std_threshold = self.config.get("outlier_std_threshold", 3.0)
    
    def validate_all(self, df: pd.DataFrame, label_col: str = "PPP_Q_Label") -> Dict[str, Any]:
        """
        Run all validation checks on dataset.
        
        Args:
            df: DataFrame to validate
            label_col: Name of label column
            
        Returns:
            Dictionary with all validation results
        """
        logger.info(f"Starting data validation on {len(df)} rows, {len(df.columns)} columns")
        
        results = {
            "data_shape": {"rows": len(df), "columns": len(df.columns)},
            "checks": {},
            "passed": True,
            "summary": {}
        }
        
        # Run all checks
        checks = [
            ("missing_values", self.check_missing_values(df)),
            ("duplicates", self.check_duplicates(df)),
            ("feature_types", self.check_feature_types(df)),
            ("value_ranges", self.check_value_ranges(df)),
            ("outliers", self.check_outliers(df)),
            ("label_distribution", self.check_label_distribution(df, label_col)),
        ]
        
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
        logger.info(f"Validation complete: {passed_count}/{len(checks)} checks passed")
        
        return results
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values in dataset"""
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100).round(2)
        
        cols_with_missing = missing_pct[missing_pct > 0].to_dict()
        total_missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        
        passed = total_missing_pct <= self.max_missing_pct
        
        return {
            "passed": passed,
            "total_missing_pct": round(total_missing_pct, 2),
            "threshold": self.max_missing_pct,
            "columns_with_missing": cols_with_missing,
            "message": f"Missing values: {total_missing_pct:.2f}% (threshold: {self.max_missing_pct}%)"
        }
    
    def check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate rows"""
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df) * 100)
        
        passed = duplicate_pct <= self.max_duplicate_pct
        
        return {
            "passed": passed,
            "duplicate_count": int(duplicate_count),
            "duplicate_pct": round(duplicate_pct, 2),
            "threshold": self.max_duplicate_pct,
            "message": f"Duplicates: {duplicate_count} rows ({duplicate_pct:.2f}%)"
        }
    
    def check_feature_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature types are as expected"""
        type_summary = df.dtypes.astype(str).value_counts().to_dict()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Expected: mostly numeric features
        numeric_ratio = len(numeric_cols) / len(df.columns)
        passed = numeric_ratio >= 0.5  # At least 50% numeric
        
        return {
            "passed": passed,
            "type_distribution": type_summary,
            "numeric_columns": len(numeric_cols),
            "non_numeric_columns": len(non_numeric_cols),
            "numeric_ratio": round(numeric_ratio, 2),
            "message": f"Feature types: {len(numeric_cols)} numeric, {len(non_numeric_cols)} non-numeric"
        }
    
    def check_value_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for valid value ranges in numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        range_issues = []
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()
            if len(col_data) == 0:
                continue
                
            # Check for infinite values
            if np.isinf(col_data).any():
                range_issues.append(f"{col}: contains infinite values")
            
            # Check for very large values
            if col_data.abs().max() > 1e10:
                range_issues.append(f"{col}: contains very large values (>{col_data.abs().max():.2e})")
        
        passed = len(range_issues) == 0
        
        return {
            "passed": passed,
            "issues": range_issues[:10],  # Limit to first 10
            "total_issues": len(range_issues),
            "message": f"Value range issues: {len(range_issues)} found"
        }
    
    def check_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using z-score method"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        outlier_summary = {}
        total_outliers = 0
        
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()
            if len(col_data) < 10:  # Skip columns with too few values
                continue
            
            mean = col_data.mean()
            std = col_data.std()
            
            if std == 0:
                continue
            
            z_scores = np.abs((col_data - mean) / std)
            outlier_count = (z_scores > self.outlier_std_threshold).sum()
            
            if outlier_count > 0:
                outlier_summary[col] = {
                    "count": int(outlier_count),
                    "pct": round(outlier_count / len(col_data) * 100, 2)
                }
                total_outliers += outlier_count
        
        # Pass if less than 5% of data points are outliers
        total_points = len(numeric_df) * len(numeric_df.columns)
        outlier_pct = (total_outliers / total_points * 100) if total_points > 0 else 0
        passed = outlier_pct < 5.0
        
        return {
            "passed": passed,
            "columns_with_outliers": len(outlier_summary),
            "total_outliers": total_outliers,
            "outlier_pct": round(outlier_pct, 2),
            "details": dict(list(outlier_summary.items())[:5]),  # Top 5
            "message": f"Outliers: {total_outliers} ({outlier_pct:.2f}%) using {self.outlier_std_threshold}σ threshold"
        }
    
    def check_label_distribution(self, df: pd.DataFrame, label_col: str) -> Dict[str, Any]:
        """Check label distribution for class imbalance"""
        if label_col not in df.columns:
            return {
                "passed": True,
                "message": f"Label column '{label_col}' not found - skipping",
                "skipped": True
            }
        
        label_counts = df[label_col].value_counts()
        label_pcts = (label_counts / len(df) * 100).round(2).to_dict()
        
        # Check for severe imbalance (any class < 5%)
        min_class_pct = min(label_pcts.values())
        passed = min_class_pct >= 5.0
        
        return {
            "passed": passed,
            "distribution": label_pcts,
            "num_classes": len(label_counts),
            "min_class_pct": min_class_pct,
            "message": f"Label distribution: {len(label_counts)} classes, min class: {min_class_pct:.1f}%"
        }
    
    def generate_report(self) -> str:
        """Generate human-readable validation report"""
        if not self.validation_results:
            return "No validation results available. Run validate_all() first."
        
        results = self.validation_results
        lines = [
            "=" * 60,
            "DATA VALIDATION REPORT",
            "=" * 60,
            f"Dataset: {results['data_shape']['rows']} rows x {results['data_shape']['columns']} columns",
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
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def run_data_validation(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Convenience function to run data validation on train/test sets.
    
    Args:
        train_path: Path to training data CSV
        test_path: Path to test data CSV
        config: Optional configuration dict
        
    Returns:
        Combined validation results
    """
    validator = DataValidator(config)
    results = {"train": None, "test": None, "overall_passed": True}
    
    if train_path and Path(train_path).exists():
        train_df = pd.read_csv(train_path)
        results["train"] = validator.validate_all(train_df)
        if not results["train"]["passed"]:
            results["overall_passed"] = False
    
    if test_path and Path(test_path).exists():
        test_df = pd.read_csv(test_path)
        results["test"] = validator.validate_all(test_df)
        if not results["test"]["passed"]:
            results["overall_passed"] = False
    
    return results
