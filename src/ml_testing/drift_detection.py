"""
Drift Detection Module

Detects data drift between training and production data:
- Feature drift (distribution changes)
- Target drift (label distribution changes)
- Concept drift (relationship changes)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects data drift between reference (training) and current (production) data.
    
    Methods:
    - Statistical tests (KS test, Chi-square)
    - Distribution comparison
    - Feature importance drift
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize drift detector with optional config"""
        self.config = config or {}
        self.drift_results: Dict[str, Any] = {}
        
        # Thresholds
        self.p_value_threshold = self.config.get("p_value_threshold", 0.05)
        self.psi_threshold = self.config.get("psi_threshold", 0.2)  # Population Stability Index
        self.drift_score_threshold = self.config.get("drift_score_threshold", 0.3)
    
    def detect_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        label_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect drift between reference and current datasets.
        
        Args:
            reference_df: Reference (training) data
            current_df: Current (production) data
            feature_cols: List of feature columns to check
            label_col: Label column name (for target drift)
            
        Returns:
            Dictionary with drift detection results
        """
        logger.info(f"Detecting drift: reference={len(reference_df)} rows, current={len(current_df)} rows")
        
        results = {
            "reference_shape": {"rows": len(reference_df), "columns": len(reference_df.columns)},
            "current_shape": {"rows": len(current_df), "columns": len(current_df.columns)},
            "feature_drift": {},
            "target_drift": None,
            "overall_drift_detected": False,
            "summary": {}
        }
        
        # Determine feature columns
        if feature_cols is None:
            feature_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
            if label_col and label_col in feature_cols:
                feature_cols.remove(label_col)
        
        # Check feature drift
        drifted_features = []
        for col in feature_cols:
            if col not in current_df.columns:
                continue
            
            drift_result = self._check_feature_drift(
                reference_df[col].dropna(),
                current_df[col].dropna(),
                col
            )
            results["feature_drift"][col] = drift_result
            
            if drift_result.get("drift_detected", False):
                drifted_features.append(col)
        
        # Check target drift if label column specified
        if label_col and label_col in reference_df.columns and label_col in current_df.columns:
            results["target_drift"] = self._check_target_drift(
                reference_df[label_col],
                current_df[label_col]
            )
        
        # Summary
        drift_ratio = len(drifted_features) / len(feature_cols) if feature_cols else 0
        results["overall_drift_detected"] = drift_ratio > self.drift_score_threshold
        
        results["summary"] = {
            "total_features_checked": len(feature_cols),
            "drifted_features": len(drifted_features),
            "drift_ratio": round(drift_ratio, 3),
            "drifted_feature_names": drifted_features[:10],  # Top 10
            "target_drift_detected": results["target_drift"].get("drift_detected", False) if results["target_drift"] else False
        }
        
        self.drift_results = results
        logger.info(f"Drift detection complete: {len(drifted_features)}/{len(feature_cols)} features drifted")
        
        return results
    
    def _check_feature_drift(
        self,
        reference_data: pd.Series,
        current_data: pd.Series,
        feature_name: str
    ) -> Dict[str, Any]:
        """Check drift for a single feature using KS test"""
        if len(reference_data) < 10 or len(current_data) < 10:
            return {
                "drift_detected": False,
                "skipped": True,
                "reason": "Insufficient data"
            }
        
        # Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(reference_data, current_data)
        
        # Population Stability Index
        psi = self._calculate_psi(reference_data, current_data)
        
        drift_detected = p_value < self.p_value_threshold or psi > self.psi_threshold
        
        return {
            "drift_detected": drift_detected,
            "ks_statistic": round(float(ks_statistic), 4),
            "p_value": round(float(p_value), 4),
            "psi": round(float(psi), 4),
            "reference_mean": round(float(reference_data.mean()), 4),
            "current_mean": round(float(current_data.mean()), 4),
            "mean_shift_pct": round((current_data.mean() - reference_data.mean()) / (reference_data.mean() + 1e-10) * 100, 2)
        }
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins from reference data
            _, bin_edges = np.histogram(reference, bins=bins)
            
            # Calculate percentages in each bin
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            cur_counts, _ = np.histogram(current, bins=bin_edges)
            
            # Normalize to percentages
            ref_pcts = ref_counts / len(reference)
            cur_pcts = cur_counts / len(current)
            
            # Avoid division by zero
            ref_pcts = np.where(ref_pcts == 0, 0.001, ref_pcts)
            cur_pcts = np.where(cur_pcts == 0, 0.001, cur_pcts)
            
            # Calculate PSI
            psi = np.sum((cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts))
            
            return abs(psi)
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0
    
    def _check_target_drift(self, reference: pd.Series, current: pd.Series) -> Dict[str, Any]:
        """Check for target distribution drift"""
        ref_dist = reference.value_counts(normalize=True).sort_index()
        cur_dist = current.value_counts(normalize=True).sort_index()
        
        # Align distributions
        all_classes = set(ref_dist.index) | set(cur_dist.index)
        ref_aligned = pd.Series({c: ref_dist.get(c, 0) for c in all_classes})
        cur_aligned = pd.Series({c: cur_dist.get(c, 0) for c in all_classes})
        
        # Chi-square test
        try:
            # Scale to counts for chi-square
            ref_counts = (ref_aligned * len(reference)).values
            cur_counts = (cur_aligned * len(current)).values
            
            chi2, p_value = stats.chisquare(cur_counts, f_exp=ref_counts * len(current) / len(reference))
            p_value = float(p_value) if not np.isnan(p_value) else 1.0
        except Exception:
            chi2, p_value = 0, 1.0
        
        drift_detected = p_value < self.p_value_threshold
        
        return {
            "drift_detected": drift_detected,
            "chi2_statistic": round(float(chi2), 4),
            "p_value": round(float(p_value), 4),
            "reference_distribution": ref_dist.to_dict(),
            "current_distribution": cur_dist.to_dict()
        }
    
    def generate_report(self) -> str:
        """Generate human-readable drift report"""
        if not self.drift_results:
            return "No drift results available. Run detect_drift() first."
        
        results = self.drift_results
        lines = [
            "=" * 60,
            "DATA DRIFT REPORT",
            "=" * 60,
            f"Reference: {results['reference_shape']['rows']} rows",
            f"Current: {results['current_shape']['rows']} rows",
            f"Overall Drift Detected: {'⚠️ YES' if results['overall_drift_detected'] else '✅ NO'}",
            "",
            "-" * 60,
            "SUMMARY:",
            "-" * 60,
            f"Features Checked: {results['summary']['total_features_checked']}",
            f"Features Drifted: {results['summary']['drifted_features']}",
            f"Drift Ratio: {results['summary']['drift_ratio']:.1%}",
            "",
        ]
        
        if results['summary']['drifted_features'] > 0:
            lines.append("Top Drifted Features:")
            for feat in results['summary']['drifted_feature_names'][:5]:
                drift_info = results['feature_drift'].get(feat, {})
                psi = drift_info.get('psi', 0)
                lines.append(f"  - {feat}: PSI={psi:.3f}")
        
        if results.get('target_drift'):
            target_drift = results['target_drift']
            status = "⚠️ DETECTED" if target_drift.get('drift_detected') else "✅ NOT DETECTED"
            lines.append(f"\nTarget Drift: {status}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def run_drift_detection(
    reference_path: str,
    current_path: str,
    feature_cols: Optional[List[str]] = None,
    label_col: Optional[str] = None,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Convenience function to run drift detection on data files.
    
    Args:
        reference_path: Path to reference data CSV
        current_path: Path to current data CSV
        feature_cols: Feature columns to check
        label_col: Label column name
        config: Optional configuration
        
    Returns:
        Drift detection results
    """
    if not Path(reference_path).exists() or not Path(current_path).exists():
        logger.warning("Data files not found for drift detection")
        return {"error": "Data files not found", "overall_drift_detected": False}
    
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)
    
    detector = DriftDetector(config)
    return detector.detect_drift(reference_df, current_df, feature_cols, label_col)
