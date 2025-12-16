"""
Pydantic Schemas for Request/Response Models
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

# ============================================================================
# ENUMS
# ============================================================================

class AssetClass(str, Enum):
    """Asset classification tiers"""
    A_PRESERVER = "A_PRESERVER"
    B_PARTIAL = "B_PARTIAL"
    C_ERODER = "C_ERODER"
    D_DESTROYER = "D_DESTROYER"

class VolatilityLevel(str, Enum):
    """Volatility assessment levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

class CyclePosition(str, Enum):
    """Market cycle positions"""
    NEAR_ATH = "NEAR_ATH"
    CORRECTION = "CORRECTION"
    VALUE_ZONE = "VALUE_ZONE"
    DEEP_VALUE = "DEEP_VALUE"

class EntrySignal(str, Enum):
    """Entry timing signals"""
    BUY = "BUY"
    CONSIDER = "CONSIDER"
    WATCH = "WATCH"
    WAIT = "WAIT"

class GrowthPotential(str, Enum):
    """Growth potential assessment"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SATURATED = "SATURATED"

class ModelType(str, Enum):
    """Model type for predictions"""
    LGBM = "lgbm"
    XGB = "xgb"
    ENSEMBLE = "ensemble"

# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class PredictionInput(BaseModel):
    """Single asset prediction input"""
    asset: str = Field(..., description="Asset name (e.g., Bitcoin, Gold, SP500)")
    horizon_years: float = Field(5, ge=0.5, le=10, description="Investment horizon in years")
    model_type: ModelType = Field(ModelType.ENSEMBLE, description="Model type: lgbm, xgb, or ensemble (default)")
    
    @validator('asset')
    def validate_asset(cls, v):
        """Validate asset name"""
        from .config import settings
        if v not in settings.AVAILABLE_ASSETS:
            raise ValueError(f"Asset must be one of: {settings.AVAILABLE_ASSETS}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "asset": "Bitcoin",
                "horizon_years": 5,
                "model_type": "ensemble"
            }
        }

class BatchPredictionInput(BaseModel):
    """Batch prediction input"""
    assets: List[str] = Field(..., description="List of asset names")
    horizon_years: float = Field(5, ge=0.5, le=10)
    
    @validator('assets')
    def validate_assets(cls, v):
        """Validate asset list"""
        from .config import settings
        invalid = [a for a in v if a not in settings.AVAILABLE_ASSETS]
        if invalid:
            raise ValueError(f"Invalid assets: {invalid}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "assets": ["Bitcoin", "Gold", "SP500"],
                "horizon_years": 5
            }
        }

class ComparisonRequest(BaseModel):
    """Multiple asset comparison input"""
    assets: List[str] = Field(..., description="List of asset names to compare (2-10)")
    horizon_years: float = Field(5, ge=0.5, le=10, description="Investment horizon in years")
    
    @validator('assets')
    def validate_assets(cls, v):
        """Validate asset list"""
        from .config import settings
        if len(v) < 2:
            raise ValueError("At least 2 assets required for comparison")
        if len(v) > 10:
            raise ValueError("Maximum 10 assets per comparison")
        invalid = [a for a in v if a not in settings.AVAILABLE_ASSETS]
        if invalid:
            raise ValueError(f"Invalid assets: {invalid}")
        return list(set(v))  # Remove duplicates
    
    class Config:
        json_schema_extra = {
            "example": {
                "assets": ["Bitcoin", "Gold", "SP500", "Apple"],
                "horizon_years": 5
            }
        }

# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class CurrentStatus(BaseModel):
    """Current market status"""
    volatility: str = Field(..., description="Volatility level with percentage")
    cycle_position: CyclePosition = Field(..., description="Position in market cycle")
    distance_from_ath: str = Field(..., description="Distance from all-time high")
    entry_signal: EntrySignal = Field(..., description="Entry timing recommendation")
    growth_potential: GrowthPotential = Field(..., description="Growth potential assessment")
    market_cap_saturation: str = Field(..., description="Market cap saturation percentage")

class Metrics(BaseModel):
    """Key performance metrics"""
    pp_multiplier_5y: float = Field(..., description="5-year purchasing power multiplier")
    sharpe_ratio_5y: float = Field(..., description="5-year Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    real_return_5y: float = Field(..., description="5-year real return percentage")

# ============================================================================
# ENHANCED RESPONSE SCHEMAS WITH COMPONENT BREAKDOWN
# ============================================================================

class PPPQComponentScores(BaseModel):
    """
    Detailed breakdown of all 7 PPP-Q scoring components
    Shows exactly how each dimension contributes to final classification
    """
    
    # Component 1: Real Purchasing Power (25% weight)
    real_purchasing_power_score: float = Field(..., ge=0, le=100, description="Can you buy MORE goods with this asset vs cash?")
    real_purchasing_power_weight: float = Field(0.25, description="Weight in final score")
    real_purchasing_power_analysis: str = Field(..., description="Human explanation of PP preservation")
    
    # Component 2: Volatility Risk (20% weight)
    volatility_risk_score: float = Field(..., ge=0, le=100, description="How stable is this asset? (Higher = more stable)")
    volatility_risk_weight: float = Field(0.20, description="Weight in final score")
    volatility_risk_analysis: str = Field(..., description="Can you stomach the price swings?")
    
    # Component 3: Market Cycle Position (15% weight)
    market_cycle_score: float = Field(..., ge=0, le=100, description="Is now a good time to buy? (Higher = better entry)")
    market_cycle_weight: float = Field(0.15, description="Weight in final score")
    market_cycle_analysis: str = Field(..., description="Where are we in the boom-bust cycle?")
    
    # Component 4: Growth Potential (15% weight)
    growth_potential_score: float = Field(..., ge=0, le=100, description="How much MORE can this grow? (Higher = more upside)")
    growth_potential_weight: float = Field(0.15, description="Weight in final score")
    growth_potential_analysis: str = Field(..., description="Is the market saturated or early-stage?")
    
    # Component 5: Consistency (10% weight)
    consistency_score: float = Field(..., ge=0, le=100, description="Reliable returns or boom-bust? (Higher = consistent)")
    consistency_weight: float = Field(0.10, description="Weight in final score")
    consistency_analysis: str = Field(..., description="Can you count on it?")
    
    # Component 6: Recovery Strength (10% weight)
    recovery_score: float = Field(..., ge=0, le=100, description="Bounces back fast from crashes? (Higher = resilient)")
    recovery_weight: float = Field(0.10, description="Weight in final score")
    recovery_analysis: str = Field(..., description="How does it handle drawdowns?")
    
    # Component 7: Risk-Adjusted Returns (5% weight)
    risk_adjusted_score: float = Field(..., ge=0, le=100, description="Quality of returns per unit of risk (Higher = better)")
    risk_adjusted_weight: float = Field(0.05, description="Weight in final score")
    risk_adjusted_analysis: str = Field(..., description="Good returns for the risk taken?")
    
    # Final Weighted Score
    final_composite_score: float = Field(..., ge=0, le=100, description="Weighted average of all 7 components")

class RealCommodityComparison(BaseModel):
    """
    Real purchasing power measured in ACTUAL goods (not CPI)
    Shows how many eggs, milk cartons, loaves of bread you can buy
    """
    
    # Eggs Purchasing Power
    eggs_current_purchasing_power: float = Field(..., description="Dozen eggs you can buy TODAY with this asset")
    eggs_1y_ago_purchasing_power: float = Field(..., description="Dozen eggs you could buy 1 YEAR AGO")
    eggs_real_return_pct: float = Field(..., description="% change in egg-buying power (1Y)")
    eggs_interpretation: str = Field(..., description="Can you buy MORE or FEWER eggs?")
    
    # Milk Purchasing Power
    milk_current_purchasing_power: float = Field(..., description="Gallons of milk you can buy TODAY")
    milk_1y_ago_purchasing_power: float = Field(..., description="Gallons of milk you could buy 1 YEAR AGO")
    milk_real_return_pct: float = Field(..., description="% change in milk-buying power (1Y)")
    milk_interpretation: str = Field(..., description="Can you buy MORE or FEWER gallons?")
    
    # Overall Commodity Basket
    commodity_basket_score: float = Field(..., ge=0, le=100, description="Overall real commodity purchasing power (0-100)")
    commodity_basket_interpretation: str = Field(..., description="Summary: Gaining or losing real purchasing power?")

class EnhancedMetrics(BaseModel):
    """Extended metrics with more context"""
    
    # Core Metrics
    pp_multiplier_5y: float = Field(..., description="5Y PP Multiplier: 1.0 = kept pace with inflation, >1.0 = beat inflation")
    pp_multiplier_1y: float = Field(..., description="1Y PP Multiplier: Recent performance")
    sharpe_ratio_5y: float = Field(..., description="Sharpe Ratio: Return per unit of risk (>1.0 = excellent)")
    max_drawdown: float = Field(..., description="Worst peak-to-trough decline (%)")
    volatility_90d: float = Field(..., description="3-month price volatility (%)")
    real_return_5y: float = Field(..., description="5Y return after inflation (%)")
    
    # Market Position
    distance_from_ath_pct: float = Field(..., description="% below all-time high (negative = below peak)")
    distance_from_ma200_pct: float = Field(..., description="% above/below 200-day average")
    days_since_ath: int = Field(..., description="Days since reaching all-time high")
    
    # Growth
    market_cap_saturation_pct: float = Field(..., description="% of total addressable market captured")
    growth_potential_multiplier: float = Field(..., description="Theoretical max growth (e.g., 5.5x = can 5x from here)")
    
    # Recovery
    recovery_strength: float = Field(..., ge=0, le=1, description="Speed of recovery from drawdowns (0-1, higher = faster)")
    consistency: float = Field(..., ge=0, le=1, description="Return consistency across horizons (0-1, higher = reliable)")

class PredictionOutput(BaseModel):
    """
    ENHANCED: Complete prediction output with detailed component breakdown
    User can see EXACTLY why an asset got its classification
    """
    
    # Basic Classification
    asset: str = Field(..., description="Asset name")
    predicted_class: AssetClass = Field(..., description="PPP-Q tier: A (best) to D (worst)")
    confidence: float = Field(..., ge=0, le=100, description="Model confidence in this prediction (%)")
    
    # COMPONENT SCORES (NEW!)
    component_scores: Optional[PPPQComponentScores] = Field(None, description="Detailed 7-component PPP-Q breakdown (optional for speed)")
    
    # REAL COMMODITY COMPARISON (NEW!)
    real_commodity_comparison: Optional[RealCommodityComparison] = Field(None, description="Actual purchasing power in eggs, milk, bread (optional for speed)")
    
    # Current Market Status
    current_status: CurrentStatus = Field(..., description="Current market cycle and signals")
    
    # Actionable Insights
    strengths: List[str] = Field(..., description="Top 3 reasons to consider this asset")
    weaknesses: List[str] = Field(..., description="Top 3 risks/concerns")
    
    # Extended Metrics
    metrics: EnhancedMetrics = Field(..., description="Comprehensive performance metrics")
    
    # Class Probabilities
    probability_breakdown: Dict[str, float] = Field(..., description="Probability of each tier (A/B/C/D)")
    
    # Metadata
    investment_horizon_years: float = Field(..., description="Investment horizon used for this prediction")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "asset": "Bitcoin",
                "predicted_class": "B_PARTIAL",
                "confidence": 68.5,
                "component_scores": {
                    "real_purchasing_power_score": 85.0,
                    "real_purchasing_power_analysis": "EXCELLENT - Can buy 3.8x more goods than cash holder",
                    "volatility_risk_score": 35.0,
                    "volatility_risk_analysis": "HIGH RISK - 42% volatility requires strong stomach",
                    "market_cycle_score": 65.0,
                    "market_cycle_analysis": "VALUE ZONE - Reasonable entry point, not overheated",
                    "growth_potential_score": 90.0,
                    "growth_potential_analysis": "HIGH UPSIDE - Only 18% market saturated",
                    "consistency_score": 45.0,
                    "consistency_analysis": "INCONSISTENT - Boom-bust cycles, not reliable",
                    "recovery_score": 70.0,
                    "recovery_analysis": "GOOD - Recovers from crashes reasonably fast",
                    "risk_adjusted_score": 75.0,
                    "risk_adjusted_analysis": "STRONG - Sharpe 1.5 = good returns for risk",
                    "final_composite_score": 68.2
                },
                "real_commodity_comparison": {
                    "eggs_current_purchasing_power": 27143.0,
                    "eggs_1y_ago_purchasing_power": 16000.0,
                    "eggs_real_return_pct": 69.6,
                    "eggs_interpretation": "✅ You can buy 69.6% MORE eggs than 1 year ago",
                    "milk_current_purchasing_power": 23750.0,
                    "milk_1y_ago_purchasing_power": 15000.0,
                    "milk_real_return_pct": 58.3,
                    "milk_interpretation": "✅ You can buy 58.3% MORE milk than 1 year ago",
                    "commodity_basket_score": 85.0,
                    "commodity_basket_interpretation": "STRONG - Your purchasing power for REAL goods is growing significantly"
                },
                "current_status": {
                    "volatility": "HIGH (42.3%)",
                    "cycle_position": "VALUE_ZONE",
                    "distance_from_ath": "-32.0%",
                    "entry_signal": "CONSIDER",
                    "growth_potential": "HIGH",
                    "market_cap_saturation": "18.0%"
                },
                "strengths": [
                    "Strong purchasing power growth (3.80x over 5Y)",
                    "High growth potential - market only 18% saturated",
                    "Good risk-adjusted returns (Sharpe: 1.5)"
                ],
                "weaknesses": [
                    "Extreme volatility (42%) - not for the faint of heart",
                    "Severe historical drawdowns (-83%) - requires patience",
                    "Inconsistent returns - boom-bust cycles"
                ],
                "metrics": {
                    "pp_multiplier_5y": 3.800,
                    "pp_multiplier_1y": 1.450,
                    "sharpe_ratio_5y": 1.500,
                    "max_drawdown": 83.2,
                    "volatility_90d": 42.3,
                    "real_return_5y": 535.7,
                    "distance_from_ath_pct": -32.0,
                    "distance_from_ma200_pct": 15.2,
                    "days_since_ath": 420,
                    "market_cap_saturation_pct": 18.0,
                    "growth_potential_multiplier": 5.5,
                    "recovery_strength": 0.75,
                    "consistency": 0.45
                },
                "probability_breakdown": {
                    "A_PRESERVER": 12.0,
                    "B_PARTIAL": 68.0,
                    "C_ERODER": 18.0,
                    "D_DESTROYER": 2.0
                },
                "investment_horizon_years": 5.0,
                "timestamp": "2024-12-15T10:30:00"
            }
        }

class BatchPredictionOutput(BaseModel):
    """Batch prediction output"""
    predictions: List[PredictionOutput]
    count: int = Field(..., description="Number of predictions")
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthCheck(BaseModel):
    """Health check response"""
    status: str = "healthy"
    service: str
    version: str
    model: str
    uptime_seconds: float

class ModelInfo(BaseModel):
    """Model information"""
    model_type: str
    num_features: int
    classes: List[str]
    performance: Dict[str, float]
    training_date: str
    best_iteration: int

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.now)