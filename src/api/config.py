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
    
    # Model Files
    LGBM_MODEL_PATH: Path = MODEL_DIR / "lgbm_model.txt"
    XGB_MODEL_PATH: Path = MODEL_DIR / "xgb_model.json"
    LABEL_ENCODER_PATH: Path = MODEL_DIR / "label_encoder.pkl"
    FEATURE_COLUMNS_PATH: Path = MODEL_DIR / "feature_columns.json"
    
    # Data Files
    TEST_DATA_PATH: Path = DATA_DIR / "test" / "pppq_test.csv"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = LOGS_DIR / "api.log"
    
    # Model Performance
    MODEL_MACRO_F1: float = 0.9543
    MODEL_ACCURACY: float = 0.9442
    MODEL_BEST_ITERATION: int = 189
    
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