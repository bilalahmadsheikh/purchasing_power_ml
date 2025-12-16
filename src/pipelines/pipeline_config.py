"""
================================================================================
PIPELINE CONFIGURATION
================================================================================
Central configuration for automated ML pipeline

Author: Bilal Ahmad Sheikh
Date: December 2024
================================================================================
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PipelineConfig:
    """Central configuration for all pipeline components"""
    
    # ========================================================================
    # DIRECTORIES
    # ========================================================================
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed" / "pppq"
    MODEL_DIR = PROJECT_ROOT / "models" / "pppq"
    REPORTS_DIR = PROJECT_ROOT / "reports" / "pppq"
    LOGS_DIR = PROJECT_ROOT / "logs"
    PIPELINES_DIR = PROJECT_ROOT / "src" / "pipelines"
    
    # ========================================================================
    # DATA FILES
    # ========================================================================
    
    RAW_DATA_PATH = RAW_DATA_DIR / "final_consolidated_dataset.csv"
    TRAIN_DATA = PROCESSED_DIR / "train" / "pppq_train.csv"
    VAL_DATA = PROCESSED_DIR / "val" / "pppq_val.csv"
    TEST_DATA = PROCESSED_DIR / "test" / "pppq_test.csv"
    FEATURES_PATH = PROCESSED_DIR / "pppq_features.json"
    
    # Model files
    LGBM_MODEL = MODEL_DIR / "lgbm_model.txt"
    XGB_MODEL = MODEL_DIR / "xgb_model.json"
    RF_MODEL = MODEL_DIR / "rf_model.pkl"
    ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
    FEATURE_COLUMNS = MODEL_DIR / "feature_columns.json"
    
    # Pipeline state
    PIPELINE_STATE = LOGS_DIR / "pipeline_state.json"
    
    # ========================================================================
    # API KEYS (from environment)
    # ========================================================================
    
    FRED_API_KEY = os.getenv('FRED_API_KEY', 'fb82293c4f0f0124456d0446d9366d24')
    
    # ========================================================================
    # SCHEDULE CONFIGURATION
    # ========================================================================
    
    PIPELINE_INTERVAL_DAYS = 15  # Run every 15 days
    FULL_RETRAIN_INTERVAL_DAYS = 90  # Full retrain every 90 days
    
    # ========================================================================
    # DATA CONFIGURATION
    # ========================================================================
    
    START_DATE = "2010-01-01"
    
    # Time splits (NO DATA LEAKAGE!)
    TRAIN_START = '2010-01-01'
    TRAIN_END = '2021-12-31'
    VAL_START = '2022-01-01'
    VAL_END = '2023-12-31'
    TEST_START = '2024-01-01'
    TEST_END = '2025-12-31'
    
    # ========================================================================
    # ASSETS
    # ========================================================================
    
    CRYPTO_ASSETS = ['Bitcoin', 'Ethereum', 'Litecoin']
    PRECIOUS_METALS = ['Gold', 'Silver']
    EQUITY_INDICES = ['SP500', 'NASDAQ', 'DowJones']
    COMMODITIES = ['Oil']
    ETFS = ['Gold_ETF', 'TreasuryBond_ETF', 'RealEstate_ETF']
    TECH_STOCKS = ['Apple', 'Microsoft', 'JPMorgan']
    
    CORE_ASSETS = (CRYPTO_ASSETS + PRECIOUS_METALS + EQUITY_INDICES + 
                   COMMODITIES + ETFS + TECH_STOCKS)
    
    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================
    
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 7,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
        'is_unbalance': True
    }
    
    XGB_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'max_depth': 7,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50
    }
    
    RF_PARAMS = {
        'n_estimators': 300,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Training parameters
    NUM_BOOST_ROUND = 500
    EARLY_STOPPING_ROUNDS = 50
    
    # ========================================================================
    # VALIDATION THRESHOLDS
    # ========================================================================
    
    MIN_IMPROVEMENT_THRESHOLD = 0.001  # Minimum F1 improvement to deploy
    MIN_DATA_SIZE = 100  # Minimum new samples before retraining
    MIN_ACCURACY = 0.70
    MIN_MACRO_F1 = 0.65
    
    # ========================================================================
    # RETRY CONFIGURATION
    # ========================================================================
    
    RETRY_ATTEMPTS = 3
    RETRY_DELAY_SECONDS = 60
    TIMEOUT_SECONDS = 3600  # 1 hour
    
    # ========================================================================
    # EMAIL NOTIFICATION CONFIGURATION (OPTIONAL)
    # ========================================================================
    
    ENABLE_NOTIFICATIONS = os.getenv('ENABLE_NOTIFICATIONS', 'false').lower() == 'true'
    NOTIFICATION_EMAIL = os.getenv('NOTIFICATION_EMAIL', 'ba8616127@gmail.com')
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'ba8616127@gmail.com')
    SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', '')  # App password for Gmail
    
    # ========================================================================
    # MLFLOW CONFIGURATION
    # ========================================================================
    
    MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlruns"
    MLFLOW_EXPERIMENT_NAME = "PPP-Q-Classification"
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    
    LOG_LEVEL = 'INFO'
    LOG_FILE = LOGS_DIR / 'pipeline.log'
    
    @classmethod
    def create_directories(cls):
        """Create all required directories"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DIR,
            cls.PROCESSED_DIR / 'train',
            cls.PROCESSED_DIR / 'val',
            cls.PROCESSED_DIR / 'test',
            cls.MODEL_DIR,
            cls.REPORTS_DIR,
            cls.LOGS_DIR,
            cls.MLFLOW_TRACKING_URI
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_end_date(cls):
        """Get current date for data collection"""
        return datetime.now().strftime("%Y-%m-%d")


# Initialize directories on import
PipelineConfig.create_directories()
