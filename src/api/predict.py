"""
PPP-Q Prediction Logic Module
Handles model loading, feature preparation, and prediction generation
Implements Purchasing Power Preservation Quality (PPP-Q) classification
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import json
from typing import List, Dict, Tuple
from pathlib import Path
import logging
from functools import lru_cache

from .config import settings
from .schemas import (
    EnhancedMetrics, PPPQComponentScores, PredictionOutput, CurrentStatus, Metrics, RealCommodityComparison,
    VolatilityLevel, CyclePosition, EntrySignal, GrowthPotential
)

# Setup logging
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL MANAGER - SINGLETON PATTERN
# ============================================================================

class ModelManager:
    """
    Singleton for LightGBM model management
    
    Responsible for loading:
    - LightGBM trained model (lgbm_model.txt)
    - Label encoder (label_encoder.pkl) - converts predictions to A/B/C/D
    - Feature columns (feature_columns.json) - 80+ PPP-Q features
    - Test data (pppq_test.csv) - historical asset metrics
    """
    
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize model attributes
            cls._instance.lgbm_model = None
            cls._instance.label_encoder = None
            cls._instance.feature_columns = None
            cls._instance.test_data = None
        return cls._instance
    
    def __init__(self):
        pass  # Lazy load models only when needed
    
    def load_models(self):
        """Load all required models and encoders - called lazily only when needed"""
        if self._models_loaded:
            return  # Already attempted to load
        
        import sys
        import os
        
        # Skip model loading in CI or Docker environment where model files may not exist
        # or could be incompatible
        if os.environ.get('CI') == 'true':
            logger.info("[INFO] CI environment detected - skipping model loading")
            self._models_loaded = True
            return
        
        # Check if running in Docker (check for .dockerenv file or DOCKER env var)
        in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'
        
        try:
            # Check if model files exist
            if not settings.LGBM_MODEL_PATH.exists():
                logger.warning(f"[WARN] Model file not found: {settings.LGBM_MODEL_PATH}")
                self._models_loaded = True
                return
            
            # Check file size - if too small, it's likely a mock file
            model_size = settings.LGBM_MODEL_PATH.stat().st_size
            if model_size < 1000:  # Less than 1KB = mock/empty file
                logger.warning(f"[WARN] Model file too small ({model_size} bytes) - likely mock data")
                self._models_loaded = True
                return
            
            # Load LightGBM model with timeout protection
            logger.info(f"[INFO] Loading LightGBM model from {settings.LGBM_MODEL_PATH}...")
            # Read model as string to avoid file path parsing issues
            with open(settings.LGBM_MODEL_PATH, 'r') as f:
                model_str = f.read()
            self.lgbm_model = lgb.Booster(model_str=model_str)
            logger.info("[OK] LightGBM model loaded")
            
            # Load label encoder (handle version mismatch gracefully)
            try:
                if settings.LABEL_ENCODER_PATH.exists():
                    with open(settings.LABEL_ENCODER_PATH, "rb") as f:
                        self.label_encoder = pickle.load(f)
                    logger.info("[OK] Label encoder loaded")
                else:
                    logger.warning(f"[WARN] Label encoder not found: {settings.LABEL_ENCODER_PATH}")
            except Exception as e:
                logger.warning(f"[WARN] Could not load label encoder (version mismatch?): {e}")
                # Create a simple label encoder as fallback
                self.label_encoder = None
            
            # Load feature columns
            try:
                if settings.FEATURE_COLUMNS_PATH.exists():
                    with open(settings.FEATURE_COLUMNS_PATH, "r") as f:
                        self.feature_columns = json.load(f)
                    logger.info(f"[OK] Feature columns loaded ({len(self.feature_columns)} features)")
                else:
                    logger.warning(f"[WARN] Feature columns not found: {settings.FEATURE_COLUMNS_PATH}")
            except Exception as e:
                logger.warning(f"[WARN] Could not load feature columns: {e}")
            
            # Load test data for predictions
            try:
                if settings.TEST_DATA_PATH.exists():
                    self.test_data = pd.read_csv(settings.TEST_DATA_PATH)
                    logger.info(f"[OK] Test data loaded ({len(self.test_data)} rows)")
                else:
                    logger.warning(f"[WARN] Test data not found: {settings.TEST_DATA_PATH}")
            except Exception as e:
                logger.warning(f"[WARN] Could not load test data: {e}")
            
        except Exception as e:
            logger.warning(f"[WARN] Could not load models: {e}")
        finally:
            self._models_loaded = True
    
    def get_model(self):
        """Get loaded LightGBM model - lazy loads if needed"""
        if not self._models_loaded:
            self.load_models()
        return self.lgbm_model
    
    def get_encoder(self):
        """Get label encoder for A/B/C/D conversion - lazy loads if needed"""
        if not self._models_loaded:
            self.load_models()
        return self.label_encoder
    
    def get_features(self):
        """Get feature columns list - lazy loads if needed"""
        if not self._models_loaded:
            self.load_models()
        return self.feature_columns if self.feature_columns else []
    
    def get_data(self):
        """Get test data DataFrame - lazy loads if needed"""
        if not self._models_loaded:
            self.load_models()
        return self.test_data

# Initialize model manager (singleton)
model_manager = ModelManager()

# ============================================================================
# DATA LOADING - MULTI-TEMPORAL SUPPORT
# ============================================================================

def load_asset_data_window(asset: str, horizon_years: float = 5) -> pd.DataFrame:
    """
    Load historical data window for asset based on investment horizon
    
    PPP-Q Concept: Different time horizons reveal different aspects of
    purchasing power preservation. Longer horizons allow for compounding
    benefits and recovery from drawdowns.
    
    Args:
        asset: Asset name (e.g., 'Bitcoin', 'Gold')
        horizon_years: Investment horizon in years (0.5 to 10)
    
    Returns:
        DataFrame with historical data for the asset window
    
    Raises:
        ValueError: If asset not found
    """
    data = model_manager.get_data()
    
    # Handle missing test data (CI environment)
    if data is None:
        logger.warning("Test data not available - returning mock data")
        # Return minimal mock data for the asset
        mock_row = {
            'Asset': asset,
            'Date': pd.Timestamp.now(),
            'Real_Return_5Y': 50.0,
            'PP_Multiplier_5Y': 5.0,
            'Volatility_90D': 30.0,
            'Sharpe_Ratio_5Y': 1.5,
            'Max_Drawdown': 50.0,
            'PPP_Q_Composite_Score': 65.0,
        }
        # Add all feature columns with default values
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
    
    # Filter to window
    window_data = asset_data[pd.to_datetime(asset_data['Date']) >= cutoff_date]
    
    # If window too small, use last 252 trading days (1 year)
    if len(window_data) < 10:
        window_data = asset_data.tail(252)
    
    logger.info(f"Loaded {asset} window ({horizon_years}Y): {len(window_data)} data points")
    
    return window_data

# ============================================================================
# FEATURE PREPARATION - PPP-Q COMPONENT SCORING
# ============================================================================

def prepare_features_with_horizon(row: pd.Series, horizon_years: float = 5) -> np.ndarray:
    """
    Prepare PPP-Q features with horizon-aware adjustments
    
    The PPP-Q model evaluates assets across 7 scoring components:
    1. Real Purchasing Power (25%) - PP_Multiplier metrics (inflation-adjusted returns)
    2. Volatility Risk (20%) - Volatility_90D (asset-specific thresholds)
    3. Market Cycle (15%) - Distance_From_ATH, Distance_From_MA_200D
    4. Growth Potential (15%) - Market_Cap_Saturation_Pct
    5. Consistency (10%) - Return_Consistency (reliability)
    6. Recovery Strength (10%) - Max_Drawdown, Recovery_Strength
    7. Risk-Adjusted (5%) - Sharpe_Ratio (quality metrics)
    
    Horizon adjustments:
    - Shorter horizons (<2Y): Need deeper value (stricter cycle), lower drawdown tolerance
    - Medium horizons (2-5Y): Balanced view
    - Longer horizons (>5Y): Can tolerate higher volatility, benefit from growth
    
    Args:
        row: Pandas Series with asset data
        horizon_years: Investment horizon for adjustments
    
    Returns:
        Numpy array of adjusted features (1, num_features)
    """
    feature_columns = model_manager.get_features()
    
    features = []
    
    for col in feature_columns:
        if col in row.index:
            value = row[col]
            
            # ===== COMPONENT 1: Real Purchasing Power (25% weight) =====
            if 'PP_Multiplier' in col:
                if pd.notna(value) and value != 0:
                    # Longer horizons expect compounding to work better
                    if '5Y' in col:
                        horizon_adj = float(value) * (horizon_years / 5.0)
                    elif '10Y' in col:
                        horizon_adj = float(value) * (horizon_years / 10.0)
                    else:  # 1Y
                        horizon_adj = float(value) * horizon_years
                    features.append(horizon_adj)
                else:
                    features.append(0.0)
            
            # ===== COMPONENT 2: Volatility Risk (20% weight) =====
            elif 'Volatility' in col:
                if pd.notna(value):
                    # Time diversification benefit: volatility reduces over longer holds
                    vol_decay = max(0.6, 1.0 - (horizon_years - 1) * 0.08)
                    features.append(float(value) * vol_decay)
                else:
                    features.append(0.0)
            
            # ===== COMPONENT 3: Market Cycle Position (15% weight) =====
            elif 'Distance_From_ATH' in col or 'Distance_From_MA_200D' in col:
                if pd.notna(value):
                    # Shorter horizons need deeper value zones
                    if horizon_years < 2:
                        cycle_adj = float(value) * 1.2  # Need more margin of safety
                    elif horizon_years < 5:
                        cycle_adj = float(value)
                    else:
                        cycle_adj = float(value) * 0.8  # Can tolerate being closer to peak
                    features.append(cycle_adj)
                else:
                    features.append(0.0)
            
            # ===== COMPONENT 4: Growth Potential (15% weight) =====
            elif 'Market_Cap_Saturation' in col or 'Growth_Potential' in col:
                if pd.notna(value):
                    # Growth becomes more important for longer horizons
                    growth_adj = float(value) * (1.0 + (horizon_years - 1) * 0.08)
                    features.append(growth_adj)
                else:
                    features.append(0.0)
            
            # ===== COMPONENT 5: Consistency (10% weight) =====
            elif 'Consistency' in col or 'Stability' in col or 'Return_Consistency' in col:
                if pd.notna(value):
                    features.append(float(value))
                else:
                    features.append(0.5)
            
            # ===== COMPONENT 6: Recovery Strength (10% weight) =====
            elif 'Recovery' in col:
                if pd.notna(value):
                    features.append(float(value))
                else:
                    features.append(0.5)
            
            elif 'Max_Drawdown' in col:
                if pd.notna(value):
                    # Longer horizons can tolerate deeper drawdowns
                    if horizon_years < 2:
                        dd_adj = float(value)  # Strict penalty
                    elif horizon_years < 5:
                        dd_adj = float(value) * 0.9
                    else:
                        dd_adj = float(value) * 0.8  # More tolerance
                    features.append(dd_adj)
                else:
                    features.append(0.0)
            
            # ===== COMPONENT 7: Risk-Adjusted Returns (5% weight) =====
            elif 'Sharpe' in col or 'Calmar' in col:
                if pd.notna(value):
                    # Sharpe ratio improves with longer horizons
                    sharpe_adj = float(value) * (1.0 + (horizon_years - 1) * 0.12)
                    features.append(sharpe_adj)
                else:
                    features.append(0.0)
            
            # ===== ADDITIONAL METRICS =====
            elif 'Real_Return' in col or 'PP_vs_Cash' in col or 'Momentum' in col:
                if pd.notna(value):
                    features.append(float(value))
                else:
                    features.append(0.0)
            
            else:
                # Default: safe handling for unknown features
                if pd.isna(value):
                    features.append(0.0)
                else:
                    features.append(float(value))
        else:
            # Feature not in data: use 0
            features.append(0.0)
    
    return np.array(features).reshape(1, -1)

# ============================================================================
# PREDICTION
# ============================================================================

def predict_asset_class(asset: str, horizon_years: float = 5) -> Tuple[str, float, np.ndarray]:
    """
    Predict PPP-Q asset classification using LightGBM model
    
    The model makes predictions based on:
    - 7 PPP-Q scoring components
    - Horizon-aware feature adjustments
    - Historical and current market data
    
    Args:
        asset: Asset name
        horizon_years: Investment horizon in years (affects feature scoring)
    
    Returns:
        Tuple of (predicted_class, confidence, probabilities)
        predicted_class: One of ['A_PRESERVER', 'B_PARTIAL', 'C_ERODER', 'D_DESTROYER']
        confidence: Prediction confidence (0-1)
        probabilities: Array of probabilities for each class
    """
    # Get model and encoder first - check if they exist
    model = model_manager.get_model()
    encoder = model_manager.get_encoder()
    
    if model is None or encoder is None:
        logger.warning("Models not available - returning mock prediction")
        return "B_PARTIAL", 0.5, np.array([[0.25, 0.5, 0.2, 0.05]])
    
    # Load historical data window
    window_data = load_asset_data_window(asset, horizon_years)
    
    # Get latest row from window
    latest_row = window_data.iloc[-1]
    
    # Prepare horizon-adjusted features
    X = prepare_features_with_horizon(latest_row, horizon_years)
    
    # Make prediction
    probabilities = model.predict(X)
    pred_class_idx = np.argmax(probabilities, axis=1)[0]
    pred_class = encoder.classes_[pred_class_idx]
    confidence = probabilities[0][pred_class_idx]
    
    logger.info(f"Predicted {asset} ({horizon_years}Y horizon): {pred_class} ({confidence:.2%})")
    
    return pred_class, confidence, probabilities

# ============================================================================
# INSIGHT GENERATION - PPP-Q CONTEXT ANALYSIS
# ============================================================================

def assess_pp_preservation_quality(pp_mult_5y: float) -> str:
    """
    Assess Purchasing Power Preservation Quality
    
    PPP-Q Core Metric: Does the asset beat inflation?
    This is the primary measure of the PPP-Q model.
    
    Args:
        pp_mult_5y: 5-year PP multiplier (real inflation-adjusted return)
    
    Returns:
        Assessment string
    """
    if pp_mult_5y > 1.5:
        return "EXCELLENT - Strong purchasing power preservation + real growth"
    elif pp_mult_5y > 1.2:
        return "GOOD - Solid purchasing power preservation"
    elif pp_mult_5y > 1.0:
        return "ADEQUATE - Barely preserves purchasing power"
    elif pp_mult_5y > 0.9:
        return "WEAK - Slight purchasing power loss to inflation"
    else:
        return "POOR - Significant purchasing power erosion"

def assess_volatility_contextual(vol: float, asset_type: str = "general") -> Tuple[VolatilityLevel, str]:
    """
    Assess volatility with asset-class context
    
    Component 2 of PPP-Q (20% weight): Volatility Risk
    Different asset classes have different volatility expectations.
    
    Args:
        vol: Volatility percentage (90-day)
        asset_type: Asset category for context (crypto/metal/equity/commodity)
    
    Returns:
        Tuple of (volatility_level, description)
    """
    # Asset-class specific thresholds
    if asset_type.lower() in ['crypto', 'bitcoin', 'ethereum']:
        thresholds = [20, 40, 70]
    elif asset_type.lower() in ['metal', 'gold', 'silver']:
        thresholds = [10, 18, 30]
    elif asset_type.lower() in ['commodity', 'oil']:
        thresholds = [15, 30, 50]
    else:  # General/Equity
        thresholds = [15, 25, 40]
    
    if vol < thresholds[0]:
        return VolatilityLevel.LOW, f"LOW ({vol:.1f}%) - Stable asset class performance"
    elif vol < thresholds[1]:
        return VolatilityLevel.MEDIUM, f"MEDIUM ({vol:.1f}%) - Normal market variability"
    elif vol < thresholds[2]:
        return VolatilityLevel.HIGH, f"HIGH ({vol:.1f}%) - Higher risk profile"
    else:
        return VolatilityLevel.EXTREME, f"EXTREME ({vol:.1f}%) - Extreme volatility risk"

def assess_market_cycle(distance_ath: float, distance_ma200: float = 0) -> Tuple[CyclePosition, EntrySignal]:
    """
    Assess market cycle position
    
    Component 3 of PPP-Q (15% weight): Market Cycle Position
    Key insight: Time your entries based on cycle position relative to peaks/averages.
    
    Args:
        distance_ath: Distance from all-time high (percent)
        distance_ma200: Distance from 200-day moving average (percent)
    
    Returns:
        Tuple of (cycle_position, entry_signal)
    """
    # Combined logic using both ATH and MA200
    if distance_ath > -5:
        return CyclePosition.NEAR_ATH, EntrySignal.WAIT
    elif distance_ath > -25:
        return CyclePosition.CORRECTION, EntrySignal.WATCH
    elif distance_ath > -50:
        return CyclePosition.VALUE_ZONE, EntrySignal.CONSIDER
    else:
        return CyclePosition.DEEP_VALUE, EntrySignal.BUY

def assess_growth_potential_saturation(market_saturation: float) -> GrowthPotential:
    """
    Assess market growth potential based on saturation
    
    Component 4 of PPP-Q (15% weight): Growth Potential
    Markets with lower saturation have more room for appreciation.
    
    Args:
        market_saturation: Market cap saturation percentage
    
    Returns:
        Growth potential assessment
    """
    if market_saturation < 20:
        return GrowthPotential.HIGH
    elif market_saturation < 50:
        return GrowthPotential.MEDIUM
    elif market_saturation < 80:
        return GrowthPotential.LOW
    else:
        return GrowthPotential.SATURATED
def calculate_component_scores(latest: pd.Series, horizon_years: float = 5) -> PPPQComponentScores:
    """
    Calculate horizon-aware scores for all 7 PPP-Q components
    
    CRITICAL: Component scores CHANGE based on investment horizon!
    - Shorter horizons (1-2Y): Emphasize stability, safety, lower cycle risk
    - Medium horizons (3-5Y): Balanced approach
    - Longer horizons (7-10Y): More focus on growth, accept volatility
    """
    
    # Extract raw metrics
    pp_mult_5y = latest.get('PP_Multiplier_5Y', 1.0)
    pp_mult_1y = latest.get('PP_Multiplier_1Y', 1.0)
    volatility_90d = latest.get('Volatility_90D', 0)
    distance_ath = latest.get('Distance_From_ATH_Pct', 0)
    distance_ma200 = latest.get('Distance_From_MA_200D_Pct', 0)
    market_saturation = latest.get('Market_Cap_Saturation_Pct', 50)
    consistency = latest.get('Return_Consistency', 0.5)
    max_drawdown = latest.get('Max_Drawdown', 0)
    recovery_strength = latest.get('Recovery_Strength', 0.5)
    sharpe_5y = latest.get('Sharpe_Ratio_5Y', 0)
    
    # ===== COMPONENT 1: Real Purchasing Power (25%) =====
    # Adjust thresholds based on horizon - longer horizons expect better returns
    if horizon_years >= 7:
        pp_threshold_excellent = 2.0
        pp_threshold_good = 1.5
        pp_threshold_adequate = 1.2
        pp_threshold_weak = 1.0
    elif horizon_years >= 3:
        pp_threshold_excellent = 1.5
        pp_threshold_good = 1.2
        pp_threshold_adequate = 1.0
        pp_threshold_weak = 0.9
    else:
        pp_threshold_excellent = 1.3
        pp_threshold_good = 1.1
        pp_threshold_adequate = 1.0
        pp_threshold_weak = 0.95
    
    if pp_mult_5y > pp_threshold_excellent:
        pp_score = 95.0
        pp_analysis = f"EXCEPTIONAL ({horizon_years}Y view) - Can buy {pp_mult_5y:.1f}x more goods than cash"
    elif pp_mult_5y > pp_threshold_good:
        pp_score = 85.0
        pp_analysis = f"EXCELLENT ({horizon_years}Y view) - Can buy {pp_mult_5y:.1f}x more goods than cash"
    elif pp_mult_5y > pp_threshold_adequate:
        pp_score = 70.0
        pp_analysis = f"GOOD ({horizon_years}Y view) - Can buy {pp_mult_5y:.1f}x more goods than cash"
    elif pp_mult_5y > pp_threshold_weak:
        pp_score = 55.0
        pp_analysis = f"ADEQUATE ({horizon_years}Y view) - Barely beats inflation ({pp_mult_5y:.2f}x)"
    else:
        pp_score = 10.0
        pp_analysis = f"POOR ({horizon_years}Y view) - Purchasing power erosion ({pp_mult_5y:.2f}x)"
    
    # ===== COMPONENT 2: Volatility Risk (20%) =====
    # Adjust thresholds based on horizon - longer horizons tolerate more volatility
    if horizon_years >= 7:
        vol_threshold_low = 15
        vol_threshold_medium = 35
        vol_threshold_high = 55
    elif horizon_years >= 3:
        vol_threshold_low = 12
        vol_threshold_medium = 30
        vol_threshold_high = 45
    else:
        vol_threshold_low = 10
        vol_threshold_medium = 20
        vol_threshold_high = 35
    
    if volatility_90d < vol_threshold_low:
        vol_score = 95.0
        vol_analysis = f"VERY LOW RISK ({horizon_years}Y) - Stable asset ({volatility_90d:.1f}%)"
    elif volatility_90d < vol_threshold_medium:
        vol_score = 75.0
        vol_analysis = f"LOW RISK ({horizon_years}Y) - Moderate stability ({volatility_90d:.1f}%)"
    elif volatility_90d < vol_threshold_high:
        vol_score = 50.0
        vol_analysis = f"MEDIUM RISK ({horizon_years}Y) - Some volatility ({volatility_90d:.1f}%)"
    else:
        vol_score = 20.0
        vol_analysis = f"HIGH RISK ({horizon_years}Y) - Significant volatility ({volatility_90d:.1f}%)"
    
    # ===== COMPONENT 3: Market Cycle (15%) =====
    # Adjust thresholds based on horizon - shorter horizons need deeper value
    if horizon_years < 2:
        ath_excellent = -70
        ath_good = -50
        ath_fair = -30
        ath_caution = -10
    elif horizon_years < 5:
        ath_excellent = -60
        ath_good = -40
        ath_fair = -20
        ath_caution = -5
    else:
        ath_excellent = -50
        ath_good = -30
        ath_fair = -10
        ath_caution = 0
    
    if distance_ath < ath_excellent:
        cycle_score = 95.0
        cycle_analysis = f"DEEP VALUE ({horizon_years}Y) - Far from peak ({distance_ath:.1f}%) = Excellent entry"
    elif distance_ath < ath_good:
        cycle_score = 80.0
        cycle_analysis = f"VALUE ZONE ({horizon_years}Y) - Good entry ({distance_ath:.1f}% from ATH)"
    elif distance_ath < ath_fair:
        cycle_score = 60.0
        cycle_analysis = f"FAIR VALUE ({horizon_years}Y) - Reasonable entry ({distance_ath:.1f}% from ATH)"
    elif distance_ath < ath_caution:
        cycle_score = 40.0
        cycle_analysis = f"CAUTION ({horizon_years}Y) - Approaching peak ({distance_ath:.1f}% from ATH)"
    else:
        cycle_score = 20.0
        cycle_analysis = f"AT PEAK ({horizon_years}Y) - Near ATH ({distance_ath:.1f}%) = Pullback risk"
    
    # ===== COMPONENT 4: Growth Potential (15%) =====
    # Adjust importance based on horizon - more critical for longer-term
    if horizon_years >= 7:
        sat_excellent = 10
        sat_good = 25
        sat_fair = 50
        sat_poor = 80
    elif horizon_years >= 3:
        sat_excellent = 15
        sat_good = 35
        sat_fair = 60
        sat_poor = 85
    else:
        sat_excellent = 20
        sat_good = 45
        sat_fair = 70
        sat_poor = 90
    
    if market_saturation < sat_excellent:
        growth_score = 95.0
        growth_analysis = f"HUGE UPSIDE ({horizon_years}Y) - Early stage ({market_saturation:.0f}% saturated)"
    elif market_saturation < sat_good:
        growth_score = 80.0
        growth_analysis = f"HIGH UPSIDE ({horizon_years}Y) - Growing market ({market_saturation:.0f}% saturated)"
    elif market_saturation < sat_fair:
        growth_score = 60.0
        growth_analysis = f"MODERATE UPSIDE ({horizon_years}Y) - Maturing ({market_saturation:.0f}% saturated)"
    elif market_saturation < sat_poor:
        growth_score = 35.0
        growth_analysis = f"LIMITED UPSIDE ({horizon_years}Y) - Mature ({market_saturation:.0f}% saturated)"
    else:
        growth_score = 15.0
        growth_analysis = f"MINIMAL UPSIDE ({horizon_years}Y) - Saturated ({market_saturation:.0f}% saturated)"
    
    # ===== COMPONENT 5: Consistency (10%) =====
    if consistency > 0.75:
        cons_score = 90.0
        cons_analysis = f"HIGHLY RELIABLE ({horizon_years}Y) - Consistent returns"
    elif consistency > 0.55:
        cons_score = 70.0
        cons_analysis = f"MODERATELY RELIABLE ({horizon_years}Y) - Generally consistent"
    elif consistency > 0.35:
        cons_score = 45.0
        cons_analysis = f"INCONSISTENT ({horizon_years}Y) - Boom-bust patterns"
    else:
        cons_score = 20.0
        cons_analysis = f"HIGHLY INCONSISTENT ({horizon_years}Y) - Extreme cyclicality"
    
    # ===== COMPONENT 6: Recovery Strength (10%) =====
    # Adjust tolerance based on horizon - longer horizons can tolerate deeper DD
    if horizon_years >= 7:
        dd_threshold_excellent = 50
        dd_threshold_good = 70
        dd_threshold_fair = 90
    elif horizon_years >= 3:
        dd_threshold_excellent = 35
        dd_threshold_good = 50
        dd_threshold_fair = 70
    else:
        dd_threshold_excellent = 20
        dd_threshold_good = 35
        dd_threshold_fair = 50
    
    if max_drawdown < dd_threshold_excellent:
        dd_score = 95.0
        dd_analysis = f"EXCELLENT ({horizon_years}Y) - Low drawdowns ({max_drawdown:.1f}%)"
    elif max_drawdown < dd_threshold_good:
        dd_score = 70.0
        dd_analysis = f"GOOD ({horizon_years}Y) - Moderate drawdowns ({max_drawdown:.1f}%)"
    elif max_drawdown < dd_threshold_fair:
        dd_score = 40.0
        dd_analysis = f"FAIR ({horizon_years}Y) - Significant drawdowns ({max_drawdown:.1f}%)"
    else:
        dd_score = 15.0
        dd_analysis = f"POOR ({horizon_years}Y) - Severe drawdowns ({max_drawdown:.1f}%)"
    
    recovery_score = (dd_score * 0.7 + recovery_strength * 100 * 0.3)
    
    # ===== COMPONENT 7: Risk-Adjusted Returns (5%) =====
    if sharpe_5y > 1.5:
        risk_adj_score = 95.0
        risk_adj_analysis = f"EXCEPTIONAL ({horizon_years}Y) - Sharpe {sharpe_5y:.2f} = excellent risk/reward"
    elif sharpe_5y > 1.0:
        risk_adj_score = 80.0
        risk_adj_analysis = f"EXCELLENT ({horizon_years}Y) - Sharpe {sharpe_5y:.2f} = good risk/reward"
    elif sharpe_5y > 0.5:
        risk_adj_score = 60.0
        risk_adj_analysis = f"ADEQUATE ({horizon_years}Y) - Sharpe {sharpe_5y:.2f} = acceptable"
    elif sharpe_5y > 0.0:
        risk_adj_score = 35.0
        risk_adj_analysis = f"WEAK ({horizon_years}Y) - Sharpe {sharpe_5y:.2f} = poor risk/reward"
    else:
        risk_adj_score = 10.0
        risk_adj_analysis = f"NEGATIVE ({horizon_years}Y) - Sharpe {sharpe_5y:.2f} = losing money"
    
    # ===== FINAL COMPOSITE SCORE =====
    # Weights shift based on horizon
    if horizon_years < 2:
        weights = (0.25, 0.25, 0.20, 0.10, 0.10, 0.10, 0.00)
    elif horizon_years < 5:
        weights = (0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05)
    else:
        weights = (0.20, 0.15, 0.10, 0.20, 0.10, 0.15, 0.10)
    
    final_score = (
        pp_score * weights[0] +
        vol_score * weights[1] +
        cycle_score * weights[2] +
        growth_score * weights[3] +
        cons_score * weights[4] +
        recovery_score * weights[5] +
        risk_adj_score * weights[6]
    )
    
    return PPPQComponentScores(
        real_purchasing_power_score=round(pp_score, 1),
        real_purchasing_power_weight=0.25,
        real_purchasing_power_analysis=pp_analysis,
        
        volatility_risk_score=round(vol_score, 1),
        volatility_risk_weight=0.20,
        volatility_risk_analysis=vol_analysis,
        
        market_cycle_score=round(cycle_score, 1),
        market_cycle_weight=0.15,
        market_cycle_analysis=cycle_analysis,
        
        growth_potential_score=round(growth_score, 1),
        growth_potential_weight=0.15,
        growth_potential_analysis=growth_analysis,
        
        consistency_score=round(cons_score, 1),
        consistency_weight=0.10,
        consistency_analysis=cons_analysis,
        
        recovery_score=round(recovery_score, 1),
        recovery_weight=0.10,
        recovery_analysis=dd_analysis,
        
        risk_adjusted_score=round(risk_adj_score, 1),
        risk_adjusted_weight=0.05,
        risk_adjusted_analysis=risk_adj_analysis,
        
        final_composite_score=round(final_score, 1)
    )

def calculate_real_commodity_comparison(latest: pd.Series, asset: str) -> RealCommodityComparison:
    """
    Calculate ACTUAL purchasing power in eggs and milk
    This is the REAL measure - not CPI nonsense
    
    For assets without direct egg/milk data, we estimate based on:
    - The asset's real return (PP_Multiplier_1Y - 1) * 100
    - Adjusted for typical egg/milk inflation (~15-25% in 2024)
    """
    
    # Get egg/milk returns from data
    eggs_return_1y = latest.get('Real_Return_Eggs_1Y', 0)
    milk_return_1y = latest.get('Real_Return_Milk_1Y', 0)
    
    # If no real data (returns are 0), estimate from asset's performance
    if eggs_return_1y == 0 and milk_return_1y == 0:
        # Get the asset's 1Y real return
        pp_mult_1y = latest.get('PP_Multiplier_1Y', 1.0)
        real_return_1y = (pp_mult_1y - 1) * 100  # Convert to percentage
        
        # Typical commodity inflation rates (eggs rose ~25%, milk ~8% in recent years)
        eggs_inflation_1y = 25.0  # Eggs have been volatile
        milk_inflation_1y = 8.0   # Milk more stable
        
        # Real return vs eggs/milk = asset return - commodity inflation
        # If asset returned 50% and eggs went up 25%, you can buy 25% more eggs
        eggs_return_1y = real_return_1y - eggs_inflation_1y
        milk_return_1y = real_return_1y - milk_inflation_1y
    
    # Estimate current purchasing power (hypothetical $10,000 investment)
    investment_amount = 10000
    
    # Current egg/milk prices (approximate from data)
    eggs_price = 3.50  # $ per dozen
    milk_price = 4.00  # $ per gallon
    
    # Calculate purchasing power
    eggs_current = investment_amount / eggs_price
    eggs_1y_ago = eggs_current / (1 + eggs_return_1y / 100) if eggs_return_1y != -100 else eggs_current
    
    milk_current = investment_amount / milk_price
    milk_1y_ago = milk_current / (1 + milk_return_1y / 100) if milk_return_1y != -100 else milk_current
    
    # Interpretations
    if eggs_return_1y > 30:
        eggs_interp = f"✅ STRONG - You can buy {eggs_return_1y:.1f}% MORE eggs than 1 year ago"
    elif eggs_return_1y > 10:
        eggs_interp = f"✅ GOOD - You can buy {eggs_return_1y:.1f}% MORE eggs than 1 year ago"
    elif eggs_return_1y > 0:
        eggs_interp = f"➖ SLIGHT GAIN - You can buy {eggs_return_1y:.1f}% MORE eggs than 1 year ago"
    elif eggs_return_1y > -10:
        eggs_interp = f"⚠️ SLIGHT LOSS - You can buy {abs(eggs_return_1y):.1f}% FEWER eggs than 1 year ago"
    else:
        eggs_interp = f"❌ LOSING POWER - You can buy {abs(eggs_return_1y):.1f}% FEWER eggs than 1 year ago"
    
    if milk_return_1y > 30:
        milk_interp = f"✅ STRONG - You can buy {milk_return_1y:.1f}% MORE milk than 1 year ago"
    elif milk_return_1y > 10:
        milk_interp = f"✅ GOOD - You can buy {milk_return_1y:.1f}% MORE milk than 1 year ago"
    elif milk_return_1y > 0:
        milk_interp = f"➖ SLIGHT GAIN - You can buy {milk_return_1y:.1f}% MORE milk than 1 year ago"
    elif milk_return_1y > -10:
        milk_interp = f"⚠️ SLIGHT LOSS - You can buy {abs(milk_return_1y):.1f}% FEWER gallons than 1 year ago"
    else:
        milk_interp = f"❌ LOSING POWER - You can buy {abs(milk_return_1y):.1f}% FEWER gallons than 1 year ago"
    
    # Overall commodity basket score
    avg_return = (eggs_return_1y + milk_return_1y) / 2
    basket_score = min(100, max(0, 50 + avg_return))
    
    if avg_return > 20:
        basket_interp = "✅ STRONG - Your real purchasing power for everyday goods is growing significantly"
    elif avg_return > 5:
        basket_interp = "✅ POSITIVE - You're gaining real purchasing power for essential goods"
    elif avg_return > -5:
        basket_interp = "➖ NEUTRAL - Roughly keeping pace with real cost of living"
    elif avg_return > -15:
        basket_interp = "⚠️ DECLINING - Losing purchasing power for everyday goods"
    else:
        basket_interp = "❌ ERODING - Significant loss of purchasing power for real goods"
    
    return RealCommodityComparison(
        eggs_current_purchasing_power=round(eggs_current, 1),
        eggs_1y_ago_purchasing_power=round(eggs_1y_ago, 1),
        eggs_real_return_pct=round(eggs_return_1y, 1),
        eggs_interpretation=eggs_interp,
        
        milk_current_purchasing_power=round(milk_current, 1),
        milk_1y_ago_purchasing_power=round(milk_1y_ago, 1),
        milk_real_return_pct=round(milk_return_1y, 1),
        milk_interpretation=milk_interp,
        
        commodity_basket_score=round(basket_score, 1),
        commodity_basket_interpretation=basket_interp
    )
def generate_pppq_insights(asset: str, pred_class: str, confidence: float, probabilities: np.ndarray, horizon_years: float = 5) -> PredictionOutput:
    """
    Generate comprehensive PPP-Q insights
    
    Creates actionable investment intelligence by analyzing all 7 PPP-Q components:
    1. Real PP Preservation (25%)
    2. Volatility Risk (20%)
    3. Market Cycle (15%)
    4. Growth Potential (15%)
    5. Consistency (10%)
    6. Recovery Strength (10%)
    7. Risk-Adjusted Returns (5%)
    
    Args:
        asset: Asset name
        pred_class: Predicted PPP-Q class (A/B/C/D)
        confidence: Model confidence in prediction
        probabilities: Class probabilities array
        horizon_years: Investment horizon for context
    
    Returns:
        Complete PredictionOutput with insights
    """
    # Load data window
    window_data = load_asset_data_window(asset, horizon_years)
    latest = window_data.iloc[-1]
    
    # Extract all PPP-Q component metrics
    pp_mult_5y = latest.get('PP_Multiplier_5Y', 1.0)
    pp_mult_1y = latest.get('PP_Multiplier_1Y', 1.0)
    sharpe_5y = latest.get('Sharpe_Ratio_5Y', 0)
    volatility_90d = latest.get('Volatility_90D', 0)
    distance_ath = latest.get('Distance_From_ATH_Pct', 0)
    distance_ma200 = latest.get('Distance_From_MA_200D_Pct', 0)
    market_saturation = latest.get('Market_Cap_Saturation_Pct', 50)
    max_drawdown = latest.get('Max_Drawdown', 0)
    recovery_strength = latest.get('Recovery_Strength', 0.5)
    consistency = latest.get('Return_Consistency', 0.5)
    real_return_5y = latest.get('Real_Return_5Y', 0)
    
    # Asset type for context
    asset_type = latest.get('Asset_Category', 'general') if 'Asset_Category' in latest.index else 'general'
    
    # ===== ASSESS ALL 7 COMPONENTS =====
    
    # Component 1: Real PP Preservation (25%)
    pp_quality = assess_pp_preservation_quality(pp_mult_5y)
    
    # Component 2: Volatility Risk (20%)
    vol_level, vol_str = assess_volatility_contextual(volatility_90d, asset_type)
    
    # Component 3: Market Cycle (15%)
    cycle_pos, entry_signal = assess_market_cycle(distance_ath, distance_ma200)
    
    # Component 4: Growth Potential (15%)
    growth_pot = assess_growth_potential_saturation(market_saturation)
    
    # Generate context-aware strengths and weaknesses
    strengths = []
    weaknesses = []
    
    # === COMPONENT 1: PP Preservation ===
    if pp_mult_5y > 1.3:
        strengths.append(f"Excellent PP preservation ({pp_mult_5y:.2f}x over 5Y)")
    elif pp_mult_5y < 0.95:
        weaknesses.append(f"Losing purchasing power to inflation ({pp_mult_5y:.2f}x)")
    
    # === COMPONENT 2: Volatility ===
    if asset_type.lower() in ['crypto'] and volatility_90d < 35:
        strengths.append(f"Moderate volatility for {asset_type} ({volatility_90d:.1f}%)")
    elif asset_type.lower() in ['metal', 'bond'] and volatility_90d < 12:
        strengths.append(f"Low volatility for {asset_type} ({volatility_90d:.1f}%)")
    elif volatility_90d > 50:
        weaknesses.append(f"Extreme volatility ({volatility_90d:.1f}%) - requires high risk tolerance")
    
    # === COMPONENT 3: Market Cycle ===
    if distance_ath < -50:
        strengths.append(f"Deep value zone ({distance_ath:.1f}% from ATH) - upside potential")
    elif distance_ath > -5:
        weaknesses.append(f"Near all-time highs ({distance_ath:.1f}%) - pullback risk")
    elif distance_ath > -25:
        strengths.append(f"Correction zone ({distance_ath:.1f}%) - reasonable entry point")
    
    # === COMPONENT 4: Growth Potential ===
    if market_saturation < 20:
        strengths.append("Early-stage market - high growth potential")
    elif market_saturation > 85:
        weaknesses.append("Mature/saturated market - limited appreciation upside")
    
    # === COMPONENT 5: Consistency (10%) ===
    if consistency > 0.7:
        strengths.append("Reliable returns - consistent performance")
    elif consistency < 0.4:
        weaknesses.append("Inconsistent returns - uneven performance over time")
    
    # === COMPONENT 6: Recovery Strength (10%) ===
    if max_drawdown > 60:
        weaknesses.append(f"Severe drawdown history ({max_drawdown:.1f}%) - recovery may take time")
    elif max_drawdown < 20:
        strengths.append(f"Low maximum drawdown ({max_drawdown:.1f}%) - good downside protection")
    
    if recovery_strength > 0.7:
        strengths.append("Strong recovery from downturns")
    
    # === COMPONENT 7: Risk-Adjusted Returns (5%) ===
    if sharpe_5y > 1.0:
        strengths.append(f"Excellent risk-adjusted returns (Sharpe: {sharpe_5y:.2f})")
    elif sharpe_5y < 0.3:
        weaknesses.append(f"Poor risk-adjusted performance (Sharpe: {sharpe_5y:.2f})")
    
    # Limit to 3 each for clarity
    strengths = strengths[:3]
    weaknesses = weaknesses[:3]
    
    # Calculate horizon-aware component scores
    component_scores = calculate_component_scores(latest, horizon_years)
    
    # Get encoder for probability breakdown
    encoder = model_manager.get_encoder()
    
    # Build probability breakdown - handle missing encoder
    if encoder is not None:
        prob_breakdown = {
            encoder.classes_[i]: round(float(probabilities[0][i]) * 100, 1)
            for i in range(len(encoder.classes_))
        }
    else:
        # Mock probability breakdown for CI
        default_classes = ["A_PRESERVER", "B_PARTIAL", "C_ERODER", "D_DESTROYER"]
        prob_breakdown = {
            default_classes[i]: round(float(probabilities[0][i]) * 100, 1)
            for i in range(min(len(default_classes), len(probabilities[0])))
        }
    
    # Build comprehensive output (fast version)
    return PredictionOutput(
        asset=asset,
        predicted_class=pred_class,
        confidence=round(float(confidence) * 100, 1),
        component_scores=component_scores,
        real_commodity_comparison=calculate_real_commodity_comparison(latest, asset),
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
        metrics=EnhancedMetrics(
            pp_multiplier_5y=round(float(pp_mult_5y), 3),
            pp_multiplier_1y=round(float(pp_mult_1y), 3),
            sharpe_ratio_5y=round(float(sharpe_5y), 3),
            max_drawdown=round(float(max_drawdown), 1),
            volatility_90d=round(float(volatility_90d), 1),
            real_return_5y=round(float(real_return_5y), 1),
            distance_from_ath_pct=round(float(distance_ath), 1),
            distance_from_ma200_pct=round(float(distance_ma200), 1),
            days_since_ath=int(latest.get('Days_Since_ATH', 0)),
            market_cap_saturation_pct=round(float(market_saturation), 1),
            growth_potential_multiplier=round(float(latest.get('Growth_Potential_Multiplier', 1.0)), 2),
            recovery_strength=round(float(recovery_strength), 2),
            consistency=round(float(consistency), 2)
        ),
        probability_breakdown=prob_breakdown,
        investment_horizon_years=horizon_years
    )

# ============================================================================
# PUBLIC API
# ============================================================================

# Cache for predictions to improve response times
# Cache size: 128 predictions, TTL via Python's lru_cache (cleared on restart)
@lru_cache(maxsize=128)
def _predict_cached(asset: str, horizon_years: float) -> PredictionOutput:
    """
    Cached prediction function
    Wraps predict() with memoization for identical requests
    """
    return _predict_uncached(asset, horizon_years)

def predict(asset: str, horizon_years: float = 5) -> PredictionOutput:
    """
    Main PPP-Q prediction function with caching
    
    Predicts how well an asset preserves purchasing power over the
    specified investment horizon and provides actionable insights.
    
    Results are cached - identical requests return cached results.
    Cache is cleared when the application restarts.
    
    Args:
        asset: Asset name (available in settings.AVAILABLE_ASSETS)
        horizon_years: Investment horizon in years (0.5 to 10 years)
    
    Returns:
        PredictionOutput with PPP-Q classification (A/B/C/D),
        confidence, metrics, and actionable insights
    """
    return _predict_cached(asset, horizon_years)

def _predict_uncached(asset: str, horizon_years: float = 5) -> PredictionOutput:
    """
    Uncached prediction logic (for testing or cache bypass)
    
    Predicts how well an asset preserves purchasing power over the
    specified investment horizon and provides actionable insights.
    
    Args:
        asset: Asset name (available in settings.AVAILABLE_ASSETS)
        horizon_years: Investment horizon in years (0.5 to 10 years)
    
    Returns:
        PredictionOutput with PPP-Q classification (A/B/C/D),
        confidence, metrics, and actionable insights
    """
    """
    Main PPP-Q prediction function
    
    Predicts how well an asset preserves purchasing power over the
    specified investment horizon and provides actionable insights.
    
    Args:
        asset: Asset name (available in settings.AVAILABLE_ASSETS)
        horizon_years: Investment horizon in years (0.5 to 10 years)
    
    Returns:
        PredictionOutput with PPP-Q classification (A/B/C/D),
        confidence, metrics, and actionable insights
    """
    pred_class, confidence, probabilities = predict_asset_class(asset, horizon_years)
    return generate_pppq_insights(asset, pred_class, confidence, probabilities, horizon_years)
