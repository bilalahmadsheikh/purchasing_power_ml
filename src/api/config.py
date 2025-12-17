"""
Configuration Management
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_TITLE: str = "PPP-Q Investment Classifier API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Purchasing Power Preservation Quality Classifier"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    WORKERS: int = 4
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    MODEL_DIR: Path = PROJECT_ROOT / "models" / "pppq"
    DATA_DIR: Path = PROJECT_ROOT / "data" / "processed" / "pppq"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # Model Files - Classification
    LGBM_CLASSIFIER_PATH: Path = MODEL_DIR / "lgbm_classifier.txt"
    XGB_CLASSIFIER_PATH: Path = MODEL_DIR / "xgb_classifier.json"
    LABEL_ENCODER_PATH: Path = MODEL_DIR / "label_encoder.pkl"
    FEATURE_COLUMNS_PATH: Path = MODEL_DIR / "feature_columns.json"
    COMPONENT_TARGETS_PATH: Path = MODEL_DIR / "component_targets.json"

    # Legacy paths (for backward compatibility)
    LGBM_MODEL_PATH: Path = MODEL_DIR / "lgbm_classifier.txt"
    XGB_MODEL_PATH: Path = MODEL_DIR / "xgb_classifier.json"

    # Component Score Models (8 regressors)
    COMPONENT_MODEL_PATHS: dict = {
        'real_pp': MODEL_DIR / "lgbm_target_real_pp_score_regressor.txt",
        'volatility': MODEL_DIR / "lgbm_target_volatility_score_regressor.txt",
        'cycle': MODEL_DIR / "lgbm_target_cycle_score_regressor.txt",
        'growth': MODEL_DIR / "lgbm_target_growth_score_regressor.txt",
        'consistency': MODEL_DIR / "lgbm_target_consistency_score_regressor.txt",
        'recovery': MODEL_DIR / "lgbm_target_recovery_score_regressor.txt",
        'risk_adjusted': MODEL_DIR / "lgbm_target_risk_adjusted_score_regressor.txt",
        'commodity': MODEL_DIR / "lgbm_target_commodity_score_regressor.txt"
    }
    
    # Data Files
    TEST_DATA_PATH: Path = DATA_DIR / "test" / "pppq_test.csv"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = LOGS_DIR / "api.log"
    
    # Model Performance (Updated with Multi-Output Models)
    MODEL_MACRO_F1: float = 0.9630  # Ensemble (LightGBM + XGBoost)
    MODEL_ACCURACY: float = 0.9524
    LGBM_MACRO_F1: float = 0.9594
    XGB_MACRO_F1: float = 0.9650
    COMPONENT_AVG_R2: float = 0.993  # Average R² for component scores
    MODEL_VERSION: str = "v2.0.0"  # Multi-output with egg/milk features
    
    # Assets
    AVAILABLE_ASSETS: List[str] = [
        "Bitcoin", "Ethereum", "Litecoin",
        "Gold", "Silver",
        "SP500", "NASDAQ", "DowJones",
        "Oil",
        "Gold_ETF", "TreasuryBond_ETF", "RealEstate_ETF",
        "Apple", "Microsoft", "JPMorgan"
    ]
    
    # Notifications
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    EMAIL_SENDER: str = os.getenv("EMAIL_SENDER", "")
    EMAIL_PASSWORD: str = os.getenv("EMAIL_PASSWORD", "")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Allow extra env vars from .env file

settings = Settings()