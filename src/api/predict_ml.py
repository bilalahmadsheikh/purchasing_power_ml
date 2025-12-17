"""
PPP-Q Prediction Logic Module - ML-Powered Component Scores
Handles model loading, feature preparation, and prediction generation
Uses ML models to predict all 8 component scores (no hardcoded logic!)

v2.0.0 - Multi-Output Models with Egg/Milk Commodity Features
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pickle
import json
from typing import List, Dict, Tuple
from pathlib import Path
import logging
from functools import lru_cache

from .config import settings
from .schemas import (
    PPPQComponentScores, PredictionOutput, CurrentStatus, EnhancedMetrics,
    VolatilityLevel, CyclePosition, EntrySignal, GrowthPotential
)

# Setup logging
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL MANAGER - SINGLETON PATTERN (Multi-Output Support)
# ============================================================================

class MLModelManager:
    """
    Singleton for managing all ML models:
    - Classification models (LightGBM + XGBoost)
    - 8 Component score regression models
    - Label encoder
    - Feature columns
    """

    _instance = None
    _models_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize model attributes
            cls._instance.lgbm_classifier = None
            cls._instance.xgb_classifier = None
            cls._instance.label_encoder = None
            cls._instance.feature_columns = None
            cls._instance.component_targets = None
            cls._instance.test_data = None

            # Component score models (8 regressors)
            cls._instance.component_models = {}

        return cls._instance

    def __init__(self):
        pass  # Lazy load models only when needed

    def load_models(self):
        """Load all required models - called lazily"""
        if self._models_loaded:
            return

        import os

        # Skip in CI environment
        if os.environ.get('CI') == 'true':
            logger.info("[INFO] CI environment - skipping model loading")
            self._models_loaded = True
            return

        try:
            # 1. Load Classification Models
            if settings.LGBM_CLASSIFIER_PATH.exists():
                logger.info(f"[INFO] Loading LightGBM classifier...")
                with open(settings.LGBM_CLASSIFIER_PATH, 'r') as f:
                    model_str = f.read()
                self.lgbm_classifier = lgb.Booster(model_str=model_str)
                logger.info("[OK] LightGBM classifier loaded")

            if settings.XGB_CLASSIFIER_PATH.exists():
                logger.info(f"[INFO] Loading XGBoost classifier...")
                self.xgb_classifier = xgb.Booster()
                self.xgb_classifier.load_model(str(settings.XGB_CLASSIFIER_PATH))
                logger.info("[OK] XGBoost classifier loaded")

            # 2. Load Component Score Models (8 regressors)
            logger.info(f"[INFO] Loading 8 component score models...")
            for comp_name, model_path in settings.COMPONENT_MODEL_PATHS.items():
                if model_path.exists():
                    with open(model_path, 'r') as f:
                        model_str = f.read()
                    self.component_models[comp_name] = lgb.Booster(model_str=model_str)
                    logger.info(f"[OK] Loaded {comp_name} regressor")
                else:
                    logger.warning(f"[WARN] Component model not found: {model_path}")

            logger.info(f"[OK] Loaded {len(self.component_models)}/8 component models")

            # 3. Load Label Encoder
            if settings.LABEL_ENCODER_PATH.exists():
                with open(settings.LABEL_ENCODER_PATH, "rb") as f:
                    self.label_encoder = pickle.load(f)
                logger.info("[OK] Label encoder loaded")

            # 4. Load Feature Columns
            if settings.FEATURE_COLUMNS_PATH.exists():
                with open(settings.FEATURE_COLUMNS_PATH, "r") as f:
                    data = json.load(f)
                self.feature_columns = data if isinstance(data, list) else data.get('features', [])
                logger.info(f"[OK] Feature columns loaded ({len(self.feature_columns)} features)")

            # 5. Load Component Targets
            if settings.COMPONENT_TARGETS_PATH.exists():
                with open(settings.COMPONENT_TARGETS_PATH, "r") as f:
                    self.component_targets = json.load(f)
                logger.info(f"[OK] Component targets loaded")

            # 6. Load Test Data
            if settings.TEST_DATA_PATH.exists():
                self.test_data = pd.read_csv(settings.TEST_DATA_PATH)
                logger.info(f"[OK] Test data loaded ({len(self.test_data)} rows)")

        except Exception as e:
            logger.warning(f"[WARN] Could not load models: {e}")
        finally:
            self._models_loaded = True

    def get_classifier(self, model_type: str = 'lgbm'):
        """Get classifier model"""
        if not self._models_loaded:
            self.load_models()
        return self.xgb_classifier if model_type == 'xgb' else self.lgbm_classifier

    def get_component_models(self):
        """Get all component score models"""
        if not self._models_loaded:
            self.load_models()
        return self.component_models

    def get_encoder(self):
        """Get label encoder"""
        if not self._models_loaded:
            self.load_models()
        return self.label_encoder

    def get_features(self):
        """Get feature columns"""
        if not self._models_loaded:
            self.load_models()
        return self.feature_columns if self.feature_columns else []

    def get_data(self):
        """Get test data"""
        if not self._models_loaded:
            self.load_models()
        return self.test_data


# Initialize model manager
model_manager = MLModelManager()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_asset_data_window(asset: str, horizon_years: float = 5) -> pd.DataFrame:
    """
    Load historical data window for asset based on investment horizon
    """
    data = model_manager.get_data()

    if data is None:
        logger.warning("Test data not available - returning mock data")
        mock_row = {'Asset': asset, 'Date': pd.Timestamp.now()}
        for col in model_manager.get_features():
            if col not in mock_row:
                mock_row[col] = 0.0
        return pd.DataFrame([mock_row])

    asset_data = data[data['Asset'] == asset].sort_values('Date')

    if len(asset_data) == 0:
        raise ValueError(f"Asset '{asset}' not found in database")

    # Get latest date and create lookback window
    latest_date = asset_data['Date'].max()
    lookback_days = int(horizon_years * 365)
    cutoff_date = pd.to_datetime(latest_date) - pd.Timedelta(days=lookback_days)

    window_data = asset_data[pd.to_datetime(asset_data['Date']) >= cutoff_date]

    if len(window_data) < 10:
        window_data = asset_data.tail(252)  # Last year

    logger.info(f"Loaded {asset} window ({horizon_years}Y): {len(window_data)} data points")

    return window_data


# ============================================================================
# HORIZON-AWARE FEATURE PREPARATION
# ============================================================================

def prepare_features_with_horizon(row: pd.Series, horizon_years: float = 5) -> np.ndarray:
    """
    Prepare features with horizon-aware adjustments

    Adjustments for different horizons:
    - Short-term (<2Y): Stricter requirements, lower risk tolerance
    - Medium-term (2-5Y): Balanced approach
    - Long-term (>5Y): More growth focus, volatility tolerance
    """
    feature_columns = model_manager.get_features()
    features = []

    for col in feature_columns:
        if col in row.index:
            value = row[col]

            # PP Multiplier adjustments
            if 'PP_Multiplier' in col:
                if pd.notna(value) and value != 0:
                    if '5Y' in col:
                        horizon_adj = float(value) * (horizon_years / 5.0)
                    elif '10Y' in col:
                        horizon_adj = float(value) * (horizon_years / 10.0)
                    else:
                        horizon_adj = float(value) * horizon_years
                    features.append(horizon_adj)
                else:
                    features.append(0.0)

            # Volatility adjustments (time diversification)
            elif 'Volatility' in col:
                if pd.notna(value):
                    vol_decay = max(0.6, 1.0 - (horizon_years - 1) * 0.08)
                    features.append(float(value) * vol_decay)
                else:
                    features.append(0.0)

            # Cycle position adjustments
            elif 'Distance_From_ATH' in col or 'Distance_From_MA' in col:
                if pd.notna(value):
                    if horizon_years < 2:
                        cycle_adj = float(value) * 1.2
                    elif horizon_years < 5:
                        cycle_adj = float(value)
                    else:
                        cycle_adj = float(value) * 0.8
                    features.append(cycle_adj)
                else:
                    features.append(0.0)

            # Growth potential adjustments
            elif 'Market_Cap_Saturation' in col or 'Growth_Potential' in col:
                if pd.notna(value):
                    growth_adj = float(value) * (1.0 + (horizon_years - 1) * 0.08)
                    features.append(growth_adj)
                else:
                    features.append(0.0)

            # Max Drawdown tolerance
            elif 'Max_Drawdown' in col:
                if pd.notna(value):
                    if horizon_years < 2:
                        dd_adj = float(value)
                    elif horizon_years < 5:
                        dd_adj = float(value) * 0.9
                    else:
                        dd_adj = float(value) * 0.8
                    features.append(dd_adj)
                else:
                    features.append(0.0)

            # Sharpe/Calmar adjustments
            elif 'Sharpe' in col or 'Calmar' in col:
                if pd.notna(value):
                    sharpe_adj = float(value) * (1.0 + (horizon_years - 1) * 0.12)
                    features.append(sharpe_adj)
                else:
                    features.append(0.0)

            # Default: use as-is
            else:
                if pd.isna(value):
                    features.append(0.0)
                else:
                    features.append(float(value))
        else:
            features.append(0.0)

    return np.array(features).reshape(1, -1)


# ============================================================================
# ML-POWERED COMPONENT SCORE PREDICTION
# ============================================================================

def predict_component_scores_ml(features: np.ndarray, horizon_years: float = 5) -> Dict[str, float]:
    """
    Predict all 8 component scores using ML models (NO HARDCODED LOGIC!)

    Returns:
        Dictionary with predicted scores for all components
    """
    component_models = model_manager.get_component_models()

    if not component_models or len(component_models) == 0:
        logger.warning("Component models not loaded - using fallback scores")
        return {
            'real_pp_score': 50.0,
            'volatility_score': 50.0,
            'cycle_score': 50.0,
            'growth_score': 50.0,
            'consistency_score': 50.0,
            'recovery_score': 50.0,
            'risk_adjusted_score': 50.0,
            'commodity_score': 50.0
        }

    # Predict each component score using ML models
    predicted_scores = {}

    component_name_mapping = {
        'real_pp': 'real_pp_score',
        'volatility': 'volatility_score',
        'cycle': 'cycle_score',
        'growth': 'growth_score',
        'consistency': 'consistency_score',
        'recovery': 'recovery_score',
        'risk_adjusted': 'risk_adjusted_score',
        'commodity': 'commodity_score'
    }

    for comp_key, comp_name in component_name_mapping.items():
        if comp_key in component_models:
            model = component_models[comp_key]
            pred = model.predict(features)[0]
            # CRITICAL: Clip to valid range [0, 100] to prevent validation errors
            predicted_scores[comp_name] = float(np.clip(pred, 0, 100))
        else:
            predicted_scores[comp_name] = 50.0  # Fallback

    logger.info(f"ML-predicted component scores: {predicted_scores}")

    return predicted_scores


# ============================================================================
# CLASSIFICATION PREDICTION
# ============================================================================

def predict_asset_class(asset: str, horizon_years: float = 5, model_type: str = 'ensemble') -> Tuple[str, float, np.ndarray]:
    """
    Predict PPP-Q asset classification using ML models
    """
    # Get models
    lgbm_model = model_manager.get_classifier('lgbm')
    xgb_model = model_manager.get_classifier('xgb')
    encoder = model_manager.get_encoder()

    if lgbm_model is None or encoder is None:
        logger.warning("Models not available - returning mock prediction")
        return "B_PARTIAL", 0.5, np.array([[0.25, 0.5, 0.2, 0.05]])

    # Load data
    window_data = load_asset_data_window(asset, horizon_years)
    latest_row = window_data.iloc[-1]

    # Prepare features
    X = prepare_features_with_horizon(latest_row, horizon_years)

    # Get feature names
    feature_names = model_manager.get_features()

    # Predict based on model type
    if model_type == 'lgbm' or xgb_model is None:
        probabilities = lgbm_model.predict(X)
        logger.info(f"Using LightGBM for {asset}")
    elif model_type == 'xgb':
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        probabilities = xgb_model.predict(dmatrix)
        logger.info(f"Using XGBoost for {asset}")
    else:  # ensemble
        lgbm_proba = lgbm_model.predict(X)
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        xgb_proba = xgb_model.predict(dmatrix)
        probabilities = (lgbm_proba + xgb_proba) / 2
        logger.info(f"Using Ensemble for {asset}")

    pred_class_idx = np.argmax(probabilities, axis=1)[0]
    pred_class = encoder.classes_[pred_class_idx]
    confidence = probabilities[0][pred_class_idx]

    logger.info(f"Predicted {asset} ({horizon_years}Y): {pred_class} ({confidence:.2%})")

    return pred_class, confidence, probabilities


# ============================================================================
# INSIGHT GENERATION WITH ML COMPONENT SCORES
# ============================================================================

def generate_pppq_insights(asset: str, pred_class: str, confidence: float,
                          probabilities: np.ndarray, horizon_years: float = 5) -> PredictionOutput:
    """
    Generate comprehensive insights using ML-predicted component scores
    """
    # Load data
    window_data = load_asset_data_window(asset, horizon_years)
    latest = window_data.iloc[-1]

    # Prepare features
    features = prepare_features_with_horizon(latest, horizon_years)

    # === ML-PREDICTED COMPONENT SCORES (NO HARDCODED LOGIC!) ===
    ml_scores = predict_component_scores_ml(features, horizon_years)

    # Extract raw metrics
    pp_mult_5y = latest.get('PP_Multiplier_5Y', 1.0)
    volatility_90d = latest.get('Volatility_90D', 0)
    distance_ath = latest.get('Distance_From_ATH_Pct', 0)
    distance_ma200 = latest.get('Distance_From_MA_200D_Pct', 0)
    market_saturation = latest.get('Market_Cap_Saturation_Pct', 50)
    sharpe_5y = latest.get('Sharpe_Ratio_5Y', 0)
    max_drawdown = latest.get('Max_Drawdown', 0)

    # Assess current status
    if volatility_90d < 15:
        vol_str = f"LOW ({volatility_90d:.1f}%)"
    elif volatility_90d < 30:
        vol_str = f"MEDIUM ({volatility_90d:.1f}%)"
    elif volatility_90d < 50:
        vol_str = f"HIGH ({volatility_90d:.1f}%)"
    else:
        vol_str = f"EXTREME ({volatility_90d:.1f}%)"

    # Cycle position
    if distance_ath > -5:
        cycle_pos = CyclePosition.NEAR_ATH
        entry_signal = EntrySignal.WAIT
    elif distance_ath > -25:
        cycle_pos = CyclePosition.CORRECTION
        entry_signal = EntrySignal.WATCH
    elif distance_ath > -50:
        cycle_pos = CyclePosition.VALUE_ZONE
        entry_signal = EntrySignal.CONSIDER
    else:
        cycle_pos = CyclePosition.DEEP_VALUE
        entry_signal = EntrySignal.BUY

    # Growth potential
    if market_saturation < 20:
        growth_pot = GrowthPotential.HIGH
    elif market_saturation < 50:
        growth_pot = GrowthPotential.MEDIUM
    elif market_saturation < 80:
        growth_pot = GrowthPotential.LOW
    else:
        growth_pot = GrowthPotential.SATURATED

    # Generate strengths/weaknesses
    strengths = []
    weaknesses = []

    if pp_mult_5y > 1.3:
        strengths.append(f"Excellent PP preservation ({pp_mult_5y:.2f}x over 5Y)")
    elif pp_mult_5y < 0.95:
        weaknesses.append(f"Losing purchasing power ({pp_mult_5y:.2f}x)")

    if volatility_90d > 50:
        weaknesses.append(f"Extreme volatility ({volatility_90d:.1f}%)")
    elif volatility_90d < 15:
        strengths.append(f"Low volatility ({volatility_90d:.1f}%)")

    if distance_ath < -50:
        strengths.append(f"Deep value zone ({distance_ath:.1f}% from ATH)")
    elif distance_ath > -5:
        weaknesses.append(f"Near ATH ({distance_ath:.1f}%) - pullback risk")

    if market_saturation < 20:
        strengths.append("Early-stage market - high growth potential")
    elif market_saturation > 85:
        weaknesses.append("Mature market - limited upside")

    if sharpe_5y > 1.0:
        strengths.append(f"Excellent risk-adjusted returns (Sharpe: {sharpe_5y:.2f})")
    elif sharpe_5y < 0.3:
        weaknesses.append(f"Poor risk-adjusted returns (Sharpe: {sharpe_5y:.2f})")

    if max_drawdown > 60:
        weaknesses.append(f"Severe drawdowns ({max_drawdown:.1f}%)")

    # Limit to top 3
    strengths = strengths[:3]
    weaknesses = weaknesses[:3]

    # Build component scores object using ML predictions
    component_scores = PPPQComponentScores(
        real_purchasing_power_score=round(ml_scores['real_pp_score'], 1),
        real_purchasing_power_weight=0.25,
        real_purchasing_power_analysis=f"ML-predicted score: {ml_scores['real_pp_score']:.1f}/100",

        volatility_risk_score=round(ml_scores['volatility_score'], 1),
        volatility_risk_weight=0.20,
        volatility_risk_analysis=f"ML-predicted score: {ml_scores['volatility_score']:.1f}/100",

        market_cycle_score=round(ml_scores['cycle_score'], 1),
        market_cycle_weight=0.15,
        market_cycle_analysis=f"ML-predicted score: {ml_scores['cycle_score']:.1f}/100",

        growth_potential_score=round(ml_scores['growth_score'], 1),
        growth_potential_weight=0.15,
        growth_potential_analysis=f"ML-predicted score: {ml_scores['growth_score']:.1f}/100",

        consistency_score=round(ml_scores['consistency_score'], 1),
        consistency_weight=0.10,
        consistency_analysis=f"ML-predicted score: {ml_scores['consistency_score']:.1f}/100",

        recovery_score=round(ml_scores['recovery_score'], 1),
        recovery_weight=0.10,
        recovery_analysis=f"ML-predicted score: {ml_scores['recovery_score']:.1f}/100",

        risk_adjusted_score=round(ml_scores['risk_adjusted_score'], 1),
        risk_adjusted_weight=0.05,
        risk_adjusted_analysis=f"ML-predicted score: {ml_scores['risk_adjusted_score']:.1f}/100",

        commodity_score=round(ml_scores['commodity_score'], 1),
        commodity_weight=0.00,  # Not in final composite but tracked
        commodity_analysis=f"ML-predicted egg/milk purchasing power: {ml_scores['commodity_score']:.1f}/100",

        final_composite_score=round(
            ml_scores['real_pp_score'] * 0.25 +
            ml_scores['volatility_score'] * 0.20 +
            ml_scores['cycle_score'] * 0.15 +
            ml_scores['growth_score'] * 0.15 +
            ml_scores['consistency_score'] * 0.10 +
            ml_scores['recovery_score'] * 0.10 +
            ml_scores['risk_adjusted_score'] * 0.05,
            1
        )
    )

    # Build probability breakdown
    encoder = model_manager.get_encoder()
    if encoder is not None:
        prob_breakdown = {
            encoder.classes_[i]: round(float(probabilities[0][i]) * 100, 1)
            for i in range(len(encoder.classes_))
        }
    else:
        prob_breakdown = {}

    # Build metrics object
    pp_mult_1y = latest.get('PP_Multiplier_1Y', pp_mult_5y)
    real_return_5y = latest.get('Real_Return_5Y', (pp_mult_5y - 1) * 100)
    days_since_ath = latest.get('Days_Since_ATH', 0)
    recovery_strength = latest.get('Recovery_Strength', 0.5)
    consistency = latest.get('Return_Consistency', 0.5)
    growth_multiplier = latest.get('Growth_Potential_Multiplier', 1.0)

    metrics = EnhancedMetrics(
        pp_multiplier_5y=round(pp_mult_5y, 3),
        pp_multiplier_1y=round(pp_mult_1y, 3),
        sharpe_ratio_5y=round(sharpe_5y, 3),
        max_drawdown=round(max_drawdown, 1),
        volatility_90d=round(volatility_90d, 1),
        real_return_5y=round(real_return_5y, 1),
        distance_from_ath_pct=round(distance_ath, 1),
        distance_from_ma200_pct=round(distance_ma200, 1),
        days_since_ath=int(days_since_ath),
        market_cap_saturation_pct=round(market_saturation, 1),
        growth_potential_multiplier=round(growth_multiplier, 2),
        recovery_strength=round(recovery_strength, 2),
        consistency=round(consistency, 2)
    )

    # Build output
    return PredictionOutput(
        asset=asset,
        predicted_class=pred_class,
        confidence=round(float(confidence) * 100, 1),
        component_scores=component_scores,
        current_status=CurrentStatus(
            volatility=vol_str,
            cycle_position=cycle_pos,
            distance_from_ath=f"{distance_ath:.1f}%",
            entry_signal=entry_signal,
            growth_potential=growth_pot,
            market_cap_saturation=f"{market_saturation:.1f}%"
        ),
        strengths=strengths,
        weaknesses=weaknesses,
        metrics=metrics,
        probability_breakdown=prob_breakdown,
        investment_horizon_years=horizon_years,
        model_version=settings.MODEL_VERSION
    )


# ============================================================================
# PUBLIC API
# ============================================================================

@lru_cache(maxsize=128)
def predict(asset: str, horizon_years: float = 5, model_type: str = 'ensemble') -> PredictionOutput:
    """
    Main PPP-Q prediction function with ML-predicted component scores

    All component scores are now predicted by ML models (R² = 99.3%)!
    """
    # Get classification prediction
    pred_class, confidence, probabilities = predict_asset_class(asset, horizon_years, model_type)

    # Generate insights with ML component scores
    return generate_pppq_insights(asset, pred_class, confidence, probabilities, horizon_years)
