"""
================================================================================
PPP-Q Investment Intelligence Dashboard - SELF-CONTAINED EDITION
================================================================================
A comprehensive Streamlit dashboard for the Purchasing Power Preservation Quotient model.

This version is FULLY SELF-CONTAINED:
- Embeds ML models directly (LightGBM + XGBoost)
- Loads data from GitHub raw URLs
- Includes data pipeline & model retraining
- No external API required - runs 100% on Streamlit Cloud

Author: Bilal Ahmad Sheikh
Institution: GIKI
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIG - MUST BE FIRST
# =============================================================================
st.set_page_config(
    page_title="PPP-Q Investment Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# GITHUB URLs for Models + GOOGLE DRIVE URLs for Data (Streamlit Cloud Ready)
# =============================================================================

# GitHub raw URLs for models (LFS-tracked files work via raw.githubusercontent.com)
GITHUB_REPO = "bilalahmadsheikh/purchasing_power_ml"
GITHUB_BRANCH = "main"

def github_raw_url(path: str) -> str:
    """Generate GitHub raw URL for a file"""
    return f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{path}"

# Model URLs from GitHub
MODEL_URLS = {
    "lgbm_model": github_raw_url("models/pppq/lgbm_model.txt"),
    "xgb_model": github_raw_url("models/pppq/xgb_model.json"),
    "feature_columns": github_raw_url("models/pppq/feature_columns.json"),
}

# Google Drive File IDs for DATA only
GDRIVE_IDS = {
    "train_data": "1nBfkQaTAfDEKICjZxy44-yDlBAEaR139",
    "test_data": "1FTZiDWO5pi06284qBq6lHc212JnAE-IH",
    "val_data": "1gACh_jmZ1I4frYnf9SGBDXnhW0gtYTm0",
    "raw_data": "1wbURisOUUhmXhgbEemUqfn1k060WpDzA",
}

# Local paths for fallback (when running locally)
import os
from pathlib import Path

# Detect if running locally vs Streamlit Cloud
IS_LOCAL = os.path.exists("data/processed/pppq/test/pppq_test.csv") or os.path.exists("../data/processed/pppq/test/pppq_test.csv")

LOCAL_PATHS = {
    "test_data": "data/processed/pppq/test/pppq_test.csv",
    "train_data": "data/processed/pppq/train/pppq_train.csv", 
    "val_data": "data/processed/pppq/val/pppq_val.csv",
    "raw_data": "data/raw/final_consolidated_dataset.csv",
    "features": "data/processed/pppq/pppq_features.json",
    "lgbm_model": "models/pppq/lgbm_model.txt",
    "xgb_model": "models/pppq/xgb_model.json",
    "feature_columns": "models/pppq/feature_columns.json",
}

# =============================================================================
# Custom CSS - FIXED VISIBILITY ISSUES
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .grade-A { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .grade-B { background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); }
    .grade-C { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); }
    .grade-D { background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); }
    
    /* FIXED: Dark text on light backgrounds for visibility */
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        color: #0d47a1;
        margin-bottom: 0.5rem;
    }
    .warning-box {
        background: #fff8e1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff8f00;
        color: #e65100;
        margin-bottom: 0.5rem;
    }
    .success-box {
        background: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #43a047;
        color: #1b5e20;
        margin-bottom: 0.5rem;
    }
    .danger-box {
        background: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #e53935;
        color: #b71c1c;
        margin-bottom: 0.5rem;
    }
    
    /* Market Regime Cards - FIXED visibility */
    .regime-card {
        background: #f5f5f5;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        color: #212121;
        margin-bottom: 1rem;
    }
    .regime-card h4 {
        color: #1565c0;
        margin-bottom: 0.5rem;
    }
    .regime-card p {
        color: #424242;
        margin: 0.3rem 0;
    }
    .regime-card strong {
        color: #1a237e;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Asset & Model Configuration (Comprehensive Asset Universe)
# =============================================================================

ASSETS = {
    # Cryptocurrencies
    "Bitcoin": {"category": "Cryptocurrency", "symbol": "BTC", "risk": "High", "liquidity": "High"},
    "Ethereum": {"category": "Cryptocurrency", "symbol": "ETH", "risk": "High", "liquidity": "High"},
    "Litecoin": {"category": "Cryptocurrency", "symbol": "LTC", "risk": "High", "liquidity": "Medium"},
    "Ripple": {"category": "Cryptocurrency", "symbol": "XRP", "risk": "Very High", "liquidity": "Medium"},
    "Cardano": {"category": "Cryptocurrency", "symbol": "ADA", "risk": "Very High", "liquidity": "Medium"},
    "Solana": {"category": "Cryptocurrency", "symbol": "SOL", "risk": "Very High", "liquidity": "Medium"},
    "Dogecoin": {"category": "Cryptocurrency", "symbol": "DOGE", "risk": "Very High", "liquidity": "Medium"},
    "Polkadot": {"category": "Cryptocurrency", "symbol": "DOT", "risk": "Very High", "liquidity": "Medium"},
    "Avalanche": {"category": "Cryptocurrency", "symbol": "AVAX", "risk": "Very High", "liquidity": "Medium"},
    "Chainlink": {"category": "Cryptocurrency", "symbol": "LINK", "risk": "Very High", "liquidity": "Medium"},
    "Polygon": {"category": "Cryptocurrency", "symbol": "MATIC", "risk": "Very High", "liquidity": "Medium"},
    "Uniswap": {"category": "Cryptocurrency", "symbol": "UNI", "risk": "Very High", "liquidity": "Medium"},
    
    # Precious Metals
    "Gold": {"category": "Precious Metal", "symbol": "XAU", "risk": "Low", "liquidity": "High"},
    "Silver": {"category": "Precious Metal", "symbol": "XAG", "risk": "Medium", "liquidity": "High"},
    "Platinum": {"category": "Precious Metal", "symbol": "XPT", "risk": "Medium", "liquidity": "Medium"},
    "Palladium": {"category": "Precious Metal", "symbol": "XPD", "risk": "Medium-High", "liquidity": "Medium"},
    "Copper": {"category": "Industrial Metal", "symbol": "HG", "risk": "Medium", "liquidity": "High"},
    
    # Major Indices
    "SP500": {"category": "Index", "symbol": "SPX", "risk": "Medium", "liquidity": "Very High"},
    "NASDAQ": {"category": "Index", "symbol": "IXIC", "risk": "Medium-High", "liquidity": "Very High"},
    "DowJones": {"category": "Index", "symbol": "DJI", "risk": "Medium", "liquidity": "Very High"},
    "Russell2000": {"category": "Index", "symbol": "RUT", "risk": "Medium-High", "liquidity": "High"},
    "VIX": {"category": "Index", "symbol": "VIX", "risk": "Very High", "liquidity": "High"},
    "FTSE100": {"category": "Index", "symbol": "FTSE", "risk": "Medium", "liquidity": "High"},
    "DAX": {"category": "Index", "symbol": "DAX", "risk": "Medium", "liquidity": "High"},
    "Nikkei225": {"category": "Index", "symbol": "N225", "risk": "Medium", "liquidity": "High"},
    "HangSeng": {"category": "Index", "symbol": "HSI", "risk": "Medium-High", "liquidity": "High"},
    
    # Tech Stocks
    "Apple": {"category": "Tech Stock", "symbol": "AAPL", "risk": "Medium", "liquidity": "Very High"},
    "Microsoft": {"category": "Tech Stock", "symbol": "MSFT", "risk": "Medium", "liquidity": "Very High"},
    "Google": {"category": "Tech Stock", "symbol": "GOOGL", "risk": "Medium", "liquidity": "Very High"},
    "Amazon": {"category": "Tech Stock", "symbol": "AMZN", "risk": "Medium", "liquidity": "Very High"},
    "Meta": {"category": "Tech Stock", "symbol": "META", "risk": "Medium-High", "liquidity": "Very High"},
    "Tesla": {"category": "Tech Stock", "symbol": "TSLA", "risk": "High", "liquidity": "Very High"},
    "Nvidia": {"category": "Tech Stock", "symbol": "NVDA", "risk": "Medium-High", "liquidity": "Very High"},
    "Netflix": {"category": "Tech Stock", "symbol": "NFLX", "risk": "Medium-High", "liquidity": "High"},
    "Adobe": {"category": "Tech Stock", "symbol": "ADBE", "risk": "Medium", "liquidity": "High"},
    "Salesforce": {"category": "Tech Stock", "symbol": "CRM", "risk": "Medium", "liquidity": "High"},
    "Intel": {"category": "Tech Stock", "symbol": "INTC", "risk": "Medium", "liquidity": "Very High"},
    "AMD": {"category": "Tech Stock", "symbol": "AMD", "risk": "Medium-High", "liquidity": "Very High"},
    "Qualcomm": {"category": "Tech Stock", "symbol": "QCOM", "risk": "Medium", "liquidity": "High"},
    "Oracle": {"category": "Tech Stock", "symbol": "ORCL", "risk": "Medium", "liquidity": "High"},
    "IBM": {"category": "Tech Stock", "symbol": "IBM", "risk": "Low-Medium", "liquidity": "High"},
    
    # Financial Stocks
    "JPMorgan": {"category": "Financial", "symbol": "JPM", "risk": "Medium", "liquidity": "Very High"},
    "BankOfAmerica": {"category": "Financial", "symbol": "BAC", "risk": "Medium", "liquidity": "Very High"},
    "WellsFargo": {"category": "Financial", "symbol": "WFC", "risk": "Medium", "liquidity": "High"},
    "GoldmanSachs": {"category": "Financial", "symbol": "GS", "risk": "Medium-High", "liquidity": "High"},
    "MorganStanley": {"category": "Financial", "symbol": "MS", "risk": "Medium-High", "liquidity": "High"},
    "Citigroup": {"category": "Financial", "symbol": "C", "risk": "Medium", "liquidity": "High"},
    "Visa": {"category": "Financial", "symbol": "V", "risk": "Medium", "liquidity": "Very High"},
    "Mastercard": {"category": "Financial", "symbol": "MA", "risk": "Medium", "liquidity": "Very High"},
    "PayPal": {"category": "Financial", "symbol": "PYPL", "risk": "Medium-High", "liquidity": "High"},
    "AmericanExpress": {"category": "Financial", "symbol": "AXP", "risk": "Medium", "liquidity": "High"},
    "Berkshire": {"category": "Financial", "symbol": "BRK-B", "risk": "Low-Medium", "liquidity": "High"},
    
    # Healthcare
    "JohnsonJohnson": {"category": "Healthcare", "symbol": "JNJ", "risk": "Low", "liquidity": "Very High"},
    "UnitedHealth": {"category": "Healthcare", "symbol": "UNH", "risk": "Low-Medium", "liquidity": "High"},
    "Pfizer": {"category": "Healthcare", "symbol": "PFE", "risk": "Low-Medium", "liquidity": "Very High"},
    "Merck": {"category": "Healthcare", "symbol": "MRK", "risk": "Low-Medium", "liquidity": "High"},
    "AbbVie": {"category": "Healthcare", "symbol": "ABBV", "risk": "Low-Medium", "liquidity": "High"},
    "Eli Lilly": {"category": "Healthcare", "symbol": "LLY", "risk": "Medium", "liquidity": "High"},
    "Moderna": {"category": "Healthcare", "symbol": "MRNA", "risk": "High", "liquidity": "High"},
    
    # Consumer/Retail
    "Walmart": {"category": "Consumer", "symbol": "WMT", "risk": "Low", "liquidity": "Very High"},
    "Costco": {"category": "Consumer", "symbol": "COST", "risk": "Low-Medium", "liquidity": "High"},
    "HomeDepot": {"category": "Consumer", "symbol": "HD", "risk": "Low-Medium", "liquidity": "High"},
    "McDonalds": {"category": "Consumer", "symbol": "MCD", "risk": "Low", "liquidity": "High"},
    "Nike": {"category": "Consumer", "symbol": "NKE", "risk": "Medium", "liquidity": "High"},
    "Starbucks": {"category": "Consumer", "symbol": "SBUX", "risk": "Medium", "liquidity": "High"},
    "CocaCola": {"category": "Consumer", "symbol": "KO", "risk": "Low", "liquidity": "Very High"},
    "PepsiCo": {"category": "Consumer", "symbol": "PEP", "risk": "Low", "liquidity": "High"},
    "Procter&Gamble": {"category": "Consumer", "symbol": "PG", "risk": "Low", "liquidity": "Very High"},
    
    # Energy
    "Oil": {"category": "Commodity", "symbol": "CL", "risk": "High", "liquidity": "High"},
    "NaturalGas": {"category": "Commodity", "symbol": "NG", "risk": "Very High", "liquidity": "High"},
    "ExxonMobil": {"category": "Energy", "symbol": "XOM", "risk": "Medium", "liquidity": "Very High"},
    "Chevron": {"category": "Energy", "symbol": "CVX", "risk": "Medium", "liquidity": "High"},
    "Shell": {"category": "Energy", "symbol": "SHEL", "risk": "Medium", "liquidity": "High"},
    "BP": {"category": "Energy", "symbol": "BP", "risk": "Medium", "liquidity": "High"},
    "ConocoPhillips": {"category": "Energy", "symbol": "COP", "risk": "Medium", "liquidity": "High"},
    
    # ETFs
    "Gold_ETF": {"category": "ETF", "symbol": "GLD", "risk": "Low", "liquidity": "Very High"},
    "Silver_ETF": {"category": "ETF", "symbol": "SLV", "risk": "Medium", "liquidity": "High"},
    "SP500_ETF": {"category": "ETF", "symbol": "SPY", "risk": "Medium", "liquidity": "Very High"},
    "Nasdaq_ETF": {"category": "ETF", "symbol": "QQQ", "risk": "Medium", "liquidity": "Very High"},
    "TreasuryBond_ETF": {"category": "ETF", "symbol": "TLT", "risk": "Low", "liquidity": "Very High"},
    "RealEstate_ETF": {"category": "ETF", "symbol": "VNQ", "risk": "Medium", "liquidity": "High"},
    "EmergingMarkets_ETF": {"category": "ETF", "symbol": "VWO", "risk": "Medium-High", "liquidity": "High"},
    "International_ETF": {"category": "ETF", "symbol": "VXUS", "risk": "Medium", "liquidity": "High"},
    "SmallCap_ETF": {"category": "ETF", "symbol": "IWM", "risk": "Medium-High", "liquidity": "High"},
    "Value_ETF": {"category": "ETF", "symbol": "VTV", "risk": "Low-Medium", "liquidity": "High"},
    "Growth_ETF": {"category": "ETF", "symbol": "VUG", "risk": "Medium", "liquidity": "High"},
    "Dividend_ETF": {"category": "ETF", "symbol": "VYM", "risk": "Low-Medium", "liquidity": "High"},
    "HighYieldBond_ETF": {"category": "ETF", "symbol": "HYG", "risk": "Medium", "liquidity": "High"},
    "CorpBond_ETF": {"category": "ETF", "symbol": "LQD", "risk": "Low-Medium", "liquidity": "High"},
    "Bitcoin_ETF": {"category": "ETF", "symbol": "IBIT", "risk": "High", "liquidity": "High"},
    "Ethereum_ETF": {"category": "ETF", "symbol": "ETHE", "risk": "High", "liquidity": "Medium"},
    
    # REITs
    "AmericanTower": {"category": "REIT", "symbol": "AMT", "risk": "Low-Medium", "liquidity": "High"},
    "Prologis": {"category": "REIT", "symbol": "PLD", "risk": "Medium", "liquidity": "High"},
    "CrownCastle": {"category": "REIT", "symbol": "CCI", "risk": "Low-Medium", "liquidity": "High"},
    "Equinix": {"category": "REIT", "symbol": "EQIX", "risk": "Medium", "liquidity": "High"},
    "SimonProperty": {"category": "REIT", "symbol": "SPG", "risk": "Medium", "liquidity": "High"},
    
    # Industrial
    "Caterpillar": {"category": "Industrial", "symbol": "CAT", "risk": "Medium", "liquidity": "High"},
    "Deere": {"category": "Industrial", "symbol": "DE", "risk": "Medium", "liquidity": "High"},
    "Boeing": {"category": "Industrial", "symbol": "BA", "risk": "Medium-High", "liquidity": "High"},
    "Honeywell": {"category": "Industrial", "symbol": "HON", "risk": "Medium", "liquidity": "High"},
    "3M": {"category": "Industrial", "symbol": "MMM", "risk": "Medium", "liquidity": "High"},
    "UPS": {"category": "Industrial", "symbol": "UPS", "risk": "Medium", "liquidity": "High"},
    
    # Telecom
    "Verizon": {"category": "Telecom", "symbol": "VZ", "risk": "Low", "liquidity": "Very High"},
    "AT&T": {"category": "Telecom", "symbol": "T", "risk": "Low", "liquidity": "Very High"},
    "T-Mobile": {"category": "Telecom", "symbol": "TMUS", "risk": "Low-Medium", "liquidity": "High"},
}

MODEL_INFO = {
    "lgbm": {"name": "LightGBM", "accuracy": "90.28%", "description": "Fast gradient boosting - Best for production"},
    "xgb": {"name": "XGBoost", "accuracy": "89.44%", "description": "Robust gradient boosting with regularization"},
    "ensemble": {"name": "Ensemble", "accuracy": "90.35%", "description": "Combined LightGBM + XGBoost for best accuracy"},
}

# PPP-Q Grade Thresholds
GRADE_THRESHOLDS = {
    'crypto': {'A': 75, 'B': 60, 'C': 40},
    'metal': {'A': 65, 'B': 50, 'C': 35},
    'default': {'A': 70, 'B': 55, 'C': 35}
}

# =============================================================================
# DATA LOADING FUNCTIONS - LOCAL FIRST, Google Drive FALLBACK (for Cloud)
# =============================================================================

# Local paths (relative to streamlit_app folder) - PRIMARY
LOCAL_PATHS_REL = {
    "test_data": "../data/processed/pppq/test/pppq_test.csv",
    "train_data": "../data/processed/pppq/train/pppq_train.csv",
    "val_data": "../data/processed/pppq/val/pppq_val.csv",
    "lgbm_model": "../models/pppq/lgbm_model.txt",
    "xgb_model": "../models/pppq/xgb_model.json",
    "feature_columns": "../models/pppq/feature_columns.json",
}

def get_base_path():
    """Get the base path for local files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return script_dir

def download_from_gdrive(file_id: str, timeout: int = 120) -> Optional[str]:
    """Download file from Google Drive with confirmation bypass for large files"""
    try:
        # Use requests session to handle cookies for large file confirmation
        session = requests.Session()
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        response = session.get(url, timeout=timeout, stream=True)
        
        # Check for virus scan confirmation page
        if 'text/html' in response.headers.get('Content-Type', ''):
            # Try to find confirmation token in cookies
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    # Retry with confirmation token
                    confirm_url = f"{url}&confirm={value}"
                    response = session.get(confirm_url, timeout=timeout)
                    break
            else:
                # Alternative: try confirm=t parameter
                confirm_url = f"{url}&confirm=t"
                response = session.get(confirm_url, timeout=timeout)
        
        if response.status_code == 200:
            content = response.text
            # Verify it's not an HTML error page
            if not content.strip().startswith('<!DOCTYPE') and not content.strip().startswith('<html'):
                return content
    except Exception as e:
        pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_csv_from_gdrive(file_id: str) -> Optional[pd.DataFrame]:
    """Load CSV directly from Google Drive"""
    try:
        content = download_from_gdrive(file_id)
        if content:
            df = pd.read_csv(io.StringIO(content))
            if len(df.columns) > 1:  # Valid CSV has multiple columns
                return df
    except Exception as e:
        pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_json_from_gdrive(file_id: str) -> Optional[Dict]:
    """Load JSON directly from Google Drive"""
    try:
        content = download_from_gdrive(file_id, timeout=30)
        if content:
            return json.loads(content)
    except:
        pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_model_from_gdrive(file_id: str) -> Optional[str]:
    """Load model file directly from Google Drive"""
    try:
        content = download_from_gdrive(file_id, timeout=120)
        if content and len(content) > 100:
            # Validate it looks like a model file (not HTML)
            first_line = content.split('\n')[0] if content else ''
            if 'tree' in first_line.lower() or first_line.startswith('{') or 'booster' in content[:500].lower():
                return content
    except:
        pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_csv_local(relative_path: str) -> Optional[pd.DataFrame]:
    """Load CSV from local file system (fallback)"""
    try:
        base = get_base_path()
        full_path = os.path.normpath(os.path.join(base, relative_path))
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            if len(df.columns) == 1 and 'version https://git-lfs.github.com/spec/v1' in str(df.columns[0]):
                return None
            return df
    except:
        pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_json_local(relative_path: str) -> Optional[Dict]:
    """Load JSON from local file system (fallback)"""
    try:
        base = get_base_path()
        full_path = os.path.normpath(os.path.join(base, relative_path))
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                content = f.read()
                if content.startswith('version https://git-lfs.github.com/spec/v1'):
                    return None
                return json.loads(content)
    except:
        pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_model_local(relative_path: str) -> Optional[str]:
    """Load model file from local file system (fallback)"""
    try:
        base = get_base_path()
        full_path = os.path.normpath(os.path.join(base, relative_path))
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                content = f.read()
                if content.startswith('version https://git-lfs.github.com/spec/v1'):
                    return None
                return content
    except:
        pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_from_github(url: str, timeout: int = 60) -> Optional[str]:
    """Load file from GitHub raw URL"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            content = response.text
            # Check it's not an HTML error or LFS pointer
            if not content.startswith('<!DOCTYPE') and not content.startswith('<html'):
                if not content.startswith('version https://git-lfs.github.com/spec/v1'):
                    return content
    except:
        pass
    return None

@st.cache_resource(show_spinner=False)
def initialize_models():
    """Initialize ML models - GitHub for MODELS, Google Drive for DATA, local fallback"""
    models = {'lgbm': None, 'xgb': None, 'feature_columns': [], 'test_data': None, 'train_data': None}
    
    # Load MODEL from GitHub first, then local fallback
    def load_model_content(key):
        # Try GitHub first
        github_url = MODEL_URLS.get(key, "")
        content = load_from_github(github_url, timeout=120) if github_url else None
        # Local fallback
        if not content or len(content) < 100:
            content = load_model_local(LOCAL_PATHS_REL.get(key, ""))
        return content
    
    # Load JSON from GitHub first, then local fallback
    def load_json_content(key):
        github_url = MODEL_URLS.get(key, "")
        content = load_from_github(github_url, timeout=30) if github_url else None
        if content:
            try:
                return json.loads(content)
            except:
                pass
        return load_json_local(LOCAL_PATHS_REL.get(key, ""))
    
    # Load DATA from Google Drive first, then local fallback
    def load_csv_content(key):
        file_id = GDRIVE_IDS.get(key, "")
        df = load_csv_from_gdrive(file_id) if file_id else None
        if df is None or len(df) == 0:
            df = load_csv_local(LOCAL_PATHS_REL.get(key, ""))
        return df
    
    # Load LightGBM model (from GitHub)
    try:
        import lightgbm as lgb
        lgbm_content = load_model_content("lgbm_model")
        if lgbm_content and len(lgbm_content) > 100:
            models['lgbm'] = lgb.Booster(model_str=lgbm_content)
    except Exception as e:
        pass
    
    # Load XGBoost model (from GitHub)
    try:
        import xgboost as xgb
        xgb_content = load_model_content("xgb_model")
        if xgb_content and len(xgb_content) > 100:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(xgb_content)
                temp_path = f.name
            models['xgb'] = xgb.Booster()
            models['xgb'].load_model(temp_path)
            os.unlink(temp_path)
    except Exception as e:
        pass
    
    # Load feature columns (from GitHub)
    feature_data = load_json_content("feature_columns")
    if feature_data:
        if isinstance(feature_data, list):
            models['feature_columns'] = feature_data
        elif isinstance(feature_data, dict):
            models['feature_columns'] = feature_data.get('features', list(feature_data.keys()) if feature_data else [])
    
    # Load ALL data (test, train, val) from Google Drive
    test_data = load_csv_content("test_data")
    train_data = load_csv_content("train_data")
    val_data = load_csv_content("val_data")
    
    # Combine all data for full history
    all_data = []
    for df in [test_data, train_data, val_data]:
        if df is not None and len(df) > 0 and 'Asset' in df.columns:
            all_data.append(df)
    
    if all_data:
        models['test_data'] = pd.concat(all_data, ignore_index=True).drop_duplicates()
        models['train_data'] = train_data
    
    return models

# =============================================================================
# PPP-Q SCORING ENGINE (Self-Contained)
# =============================================================================

def get_asset_category(asset: str) -> str:
    """Get asset category for scoring adjustments"""
    # All cryptocurrencies (including new ones)
    crypto = ['Bitcoin', 'Ethereum', 'Litecoin', 'Cardano', 'Solana', 'Ripple', 
              'Dogecoin', 'Polkadot', 'Avalanche', 'Chainlink', 'Polygon', 'Uniswap',
              'Bitcoin_ETF', 'Ethereum_ETF']
    # All precious/industrial metals
    metals = ['Gold', 'Silver', 'Platinum', 'Palladium', 'Copper', 'Gold_ETF', 'Silver_ETF']
    
    if asset in crypto:
        return 'crypto'
    elif asset in metals:
        return 'metal'
    return 'default'

def calculate_component_scores(row: pd.Series, asset: str, horizon_years: int = 5) -> Dict:
    """Calculate 7-component PPP-Q scores with HORIZON AWARENESS"""
    
    category = get_asset_category(asset)
    
    # Select PP multiplier based on horizon
    if horizon_years <= 1:
        pp_mult = row.get('PP_Multiplier_1Y', row.get('PP_Multiplier_5Y', 1.0))
        horizon_factor = 0.7  # Short-term more conservative
    elif horizon_years <= 3:
        pp_mult = row.get('PP_Multiplier_5Y', 1.0) ** (horizon_years / 5.0)
        horizon_factor = 0.85
    elif horizon_years <= 5:
        pp_mult = row.get('PP_Multiplier_5Y', 1.0)
        horizon_factor = 1.0
    else:  # Long-term (6-10 years)
        pp_mult_5y = row.get('PP_Multiplier_5Y', 1.0)
        pp_mult_10y = row.get('PP_Multiplier_10Y', pp_mult_5y ** 2)
        pp_mult = pp_mult_5y ** (horizon_years / 5.0)
        horizon_factor = 1.15  # Long-term bonus
    
    # Component 1: Real Purchasing Power (25%) - HORIZON ADJUSTED
    if pp_mult < 0.85:
        pp_score = 0.0
    elif pp_mult < 1.0:
        pp_score = 20.0 + (pp_mult - 0.85) / 0.15 * 30.0
    elif pp_mult < 1.3:
        pp_score = 50.0 + (pp_mult - 1.0) / 0.3 * 30.0
    elif pp_mult < 2.0:
        pp_score = 80.0 + (pp_mult - 1.3) / 0.7 * 15.0
    else:
        pp_score = min(100.0, 95.0 + np.log10(pp_mult - 2.0 + 1) * 5.0)
    
    pp_score = pp_score * horizon_factor  # Apply horizon adjustment
    pp_score = min(100, pp_score)  # Cap at 100
    
    # Component 2: Volatility Risk (20%) - HORIZON ADJUSTED
    volatility = row.get('Volatility_90D', 30)
    vol_multiplier = {'crypto': 2.0, 'metal': 0.8, 'default': 1.0}.get(category, 1.0)
    
    # Volatility matters LESS for longer horizons (time diversification)
    vol_decay = max(0.5, 1.0 - (horizon_years - 1) * 0.08)
    adjusted_vol = volatility * vol_multiplier * vol_decay
    
    if adjusted_vol < 10:
        vol_score = 100.0
    elif adjusted_vol < 15:
        vol_score = 90.0
    elif adjusted_vol < 25:
        vol_score = 70.0
    elif adjusted_vol < 40:
        vol_score = 45.0
    elif adjusted_vol < 60:
        vol_score = 20.0
    else:
        vol_score = max(0.0, 10.0 - (adjusted_vol - 60) / 10.0)
    
    # Component 3: Market Cycle Position (15%) - HORIZON ADJUSTED
    distance_ath = row.get('Distance_From_ATH_Pct', 0)
    
    # Short horizons need DEEPER value (stricter)
    if horizon_years <= 2:
        if distance_ath > -10:
            ath_score = 20.0  # Very risky for short-term
        elif distance_ath > -30:
            ath_score = 50.0
        elif distance_ath > -50:
            ath_score = 80.0
        else:
            ath_score = 100.0
    else:  # Longer horizons more forgiving
        if distance_ath > -5:
            ath_score = 30.0
        elif distance_ath > -20:
            ath_score = 60.0
        elif distance_ath > -50:
            ath_score = 85.0
        else:
            ath_score = 100.0
    
    distance_ma200 = row.get('Distance_From_MA_200D_Pct', 0)
    if distance_ma200 > 20:
        ma_score = 40.0
    elif distance_ma200 > 0:
        ma_score = 80.0
    elif distance_ma200 > -20:
        ma_score = 60.0
    else:
        ma_score = 30.0
    cycle_score = ath_score * 0.6 + ma_score * 0.4
    
    # Component 4: Growth Potential (15%) - MORE IMPORTANT FOR LONGER HORIZONS
    saturation = row.get('Market_Cap_Saturation_Pct', 50)
    if saturation < 10:
        growth_score = 100.0
    elif saturation < 30:
        growth_score = 85.0
    elif saturation < 50:
        growth_score = 65.0
    elif saturation < 70:
        growth_score = 45.0
    elif saturation < 90:
        growth_score = 25.0
    else:
        growth_score = 10.0
    
    # Growth matters more for longer horizons
    if horizon_years >= 7:
        growth_score = growth_score * 1.2
    growth_score = min(100, growth_score)
    
    # Component 5: Consistency (10%)
    consistency = row.get('Return_Consistency', 0.5) * 50 + row.get('PP_Stability_Index', 0.5) * 50
    
    # Component 6: Recovery (10%) - MORE IMPORTANT FOR SHORTER HORIZONS
    max_dd = row.get('Max_Drawdown', 30)
    recovery = row.get('Recovery_Strength', 0.5)
    if max_dd < 10:
        dd_score = 60.0
    elif max_dd < 25:
        dd_score = 50.0
    elif max_dd < 50:
        dd_score = 35.0
    elif max_dd < 75:
        dd_score = 15.0
    else:
        dd_score = 0.0
    
    # Drawdown matters more for short-term
    if horizon_years <= 2:
        dd_score = dd_score * 1.3
    dd_score = min(100, dd_score)
    recovery_score = dd_score * 0.6 + recovery * 100 * 0.4
    
    # Component 7: Risk-Adjusted (5%)
    sharpe = row.get('Sharpe_Ratio_5Y', 0)
    if sharpe < 0:
        risk_adj_score = 0.0
    elif sharpe < 0.5:
        risk_adj_score = sharpe / 0.5 * 50.0
    elif sharpe < 1.0:
        risk_adj_score = 50.0 + (sharpe - 0.5) / 0.5 * 30.0
    else:
        risk_adj_score = min(100.0, 80.0 + sharpe * 10.0)
    
    # Final Composite with HORIZON-ADJUSTED WEIGHTS
    if horizon_years <= 2:  # Short-term: safety focus
        weights = {'pp': 0.20, 'vol': 0.25, 'cycle': 0.20, 'growth': 0.10, 'cons': 0.10, 'rec': 0.10, 'risk': 0.05}
    elif horizon_years <= 5:  # Medium-term: balanced
        weights = {'pp': 0.25, 'vol': 0.20, 'cycle': 0.15, 'growth': 0.15, 'cons': 0.10, 'rec': 0.10, 'risk': 0.05}
    else:  # Long-term: growth focus
        weights = {'pp': 0.25, 'vol': 0.15, 'cycle': 0.10, 'growth': 0.25, 'cons': 0.10, 'rec': 0.10, 'risk': 0.05}
    
    final_score = (
        pp_score * weights['pp'] +
        vol_score * weights['vol'] +
        cycle_score * weights['cycle'] +
        growth_score * weights['growth'] +
        consistency * weights['cons'] +
        recovery_score * weights['rec'] +
        risk_adj_score * weights['risk']
    )
    
    return {
        'real_purchasing_power_score': round(pp_score, 1),
        'volatility_risk_score': round(vol_score, 1),
        'market_cycle_score': round(cycle_score, 1),
        'growth_potential_score': round(growth_score, 1),
        'consistency_score': round(consistency, 1),
        'recovery_score': round(recovery_score, 1),
        'risk_adjusted_score': round(risk_adj_score, 1),
        'final_composite_score': round(final_score, 1),
        'horizon_years': horizon_years,
        'pp_multiplier_used': round(pp_mult, 3)
    }

def assign_grade(score: float, category: str) -> str:
    """Assign PPP-Q grade based on score and asset category"""
    thresholds = GRADE_THRESHOLDS.get(category, GRADE_THRESHOLDS['default'])
    if score >= thresholds['A']:
        return 'A'
    elif score >= thresholds['B']:
        return 'B'
    elif score >= thresholds['C']:
        return 'C'
    return 'D'

def generate_insights(row: pd.Series, score: float, grade: str) -> Tuple[List[str], List[str]]:
    """Generate strengths and weaknesses based on metrics"""
    strengths = []
    weaknesses = []
    
    pp_mult = row.get('PP_Multiplier_5Y', 1.0)
    if pp_mult > 1.5:
        strengths.append(f"💰 Strong purchasing power growth ({pp_mult:.2f}x over 5Y)")
    elif pp_mult < 1.0:
        weaknesses.append(f"📉 Losing purchasing power ({pp_mult:.2f}x over 5Y)")
    
    sharpe = row.get('Sharpe_Ratio_5Y', 0)
    if sharpe > 1.0:
        strengths.append(f"📈 Excellent risk-adjusted returns (Sharpe: {sharpe:.2f})")
    elif sharpe < 0.3:
        weaknesses.append(f"⚠️ Poor risk-adjusted returns (Sharpe: {sharpe:.2f})")
    
    max_dd = row.get('Max_Drawdown', 0)
    if max_dd > 50:
        weaknesses.append(f"🔻 Severe historical drawdowns ({max_dd:.1f}%)")
    elif max_dd < 20:
        strengths.append(f"🛡️ Low drawdown risk ({max_dd:.1f}% max)")
    
    volatility = row.get('Volatility_90D', 30)
    if volatility < 15:
        strengths.append("📊 Low volatility - stable asset")
    elif volatility > 50:
        weaknesses.append(f"🌊 Extreme volatility ({volatility:.1f}%)")
    
    distance_ath = row.get('Distance_From_ATH_Pct', 0)
    if distance_ath < -50:
        strengths.append("💎 Deep value - far from ATH (opportunity)")
    elif distance_ath > -5:
        weaknesses.append("⚡ Near all-time high (pullback risk)")
    
    saturation = row.get('Market_Cap_Saturation_Pct', 50)
    if saturation < 30:
        strengths.append("🚀 High growth potential (low market saturation)")
    elif saturation > 80:
        weaknesses.append("🏔️ Limited upside (market saturated)")
    
    return strengths[:4], weaknesses[:4]

def get_current_status(row: pd.Series) -> Dict:
    """Get current market status indicators"""
    volatility = row.get('Volatility_90D', 30)
    if volatility < 15:
        vol_level = 'LOW'
    elif volatility < 30:
        vol_level = 'MEDIUM'
    elif volatility < 50:
        vol_level = 'HIGH'
    else:
        vol_level = 'EXTREME'
    
    distance_ath = row.get('Distance_From_ATH_Pct', 0)
    if distance_ath > -10:
        cycle_pos = 'NEAR_ATH'
        entry_signal = 'WAIT'
    elif distance_ath > -30:
        cycle_pos = 'CORRECTION'
        entry_signal = 'WATCH'
    elif distance_ath > -50:
        cycle_pos = 'VALUE_ZONE'
        entry_signal = 'CONSIDER'
    else:
        cycle_pos = 'DEEP_VALUE'
        entry_signal = 'BUY'
    
    return {
        'volatility': vol_level,
        'volatility_value': round(volatility, 1),
        'cycle_position': cycle_pos,
        'distance_from_ath': f"{distance_ath:.1f}%",
        'entry_signal': entry_signal
    }

def calculate_real_commodity_comparison(row: pd.Series, asset: str, horizon_years: int = 5) -> Dict:
    """
    Calculate REAL purchasing power in terms of actual commodities (eggs, milk, bread, gas)
    This shows what your $100 investment can ACTUALLY buy in real goods.
    
    WEIGHTS for Real PP Index (based on typical household consumption):
    - Eggs: 25% (protein staple)
    - Milk: 25% (dairy staple)  
    - Bread: 25% (grain staple)
    - Gasoline: 25% (energy/transport)
    """
    
    # Current commodity prices (Dec 2024 estimates)
    CURRENT_PRICES = {
        'eggs_dozen': 4.50,      # $4.50/dozen
        'milk_gallon': 4.20,     # $4.20/gallon
        'bread_loaf': 3.50,      # $3.50/loaf
        'gas_gallon': 3.20       # $3.20/gallon
    }
    
    # Prices 1 year ago (Dec 2023)
    PRICES_1Y_AGO = {
        'eggs_dozen': 4.00,
        'milk_gallon': 3.80,
        'bread_loaf': 3.20,
        'gas_gallon': 3.50
    }
    
    # Prices 5 years ago (Dec 2019)
    PRICES_5Y_AGO = {
        'eggs_dozen': 2.50,
        'milk_gallon': 3.20,
        'bread_loaf': 2.50,
        'gas_gallon': 2.60
    }
    
    # Get PP multiplier based on horizon
    if horizon_years <= 1:
        pp_mult = row.get('PP_Multiplier_1Y', 1.0)
        past_prices = PRICES_1Y_AGO
        period_label = "1 Year Ago"
    else:
        pp_mult = row.get('PP_Multiplier_5Y', 1.0)
        past_prices = PRICES_5Y_AGO
        period_label = "5 Years Ago"
    
    # Initial investment
    initial_investment = 100.0
    
    # What your investment is worth now (after asset returns)
    current_value = initial_investment * pp_mult
    
    # Calculate purchasing power for EACH commodity
    commodities = {}
    
    # EGGS
    eggs_now = current_value / CURRENT_PRICES['eggs_dozen']
    eggs_past = initial_investment / past_prices['eggs_dozen']
    eggs_change = ((eggs_now - eggs_past) / eggs_past * 100) if eggs_past > 0 else 0
    commodities['eggs'] = {
        'current': round(eggs_now, 1),
        'past': round(eggs_past, 1),
        'change_pct': round(eggs_change, 1),
        'unit': 'dozen',
        'icon': '🥚'
    }
    
    # MILK
    milk_now = current_value / CURRENT_PRICES['milk_gallon']
    milk_past = initial_investment / past_prices['milk_gallon']
    milk_change = ((milk_now - milk_past) / milk_past * 100) if milk_past > 0 else 0
    commodities['milk'] = {
        'current': round(milk_now, 1),
        'past': round(milk_past, 1),
        'change_pct': round(milk_change, 1),
        'unit': 'gallons',
        'icon': '🥛'
    }
    
    # BREAD
    bread_now = current_value / CURRENT_PRICES['bread_loaf']
    bread_past = initial_investment / past_prices['bread_loaf']
    bread_change = ((bread_now - bread_past) / bread_past * 100) if bread_past > 0 else 0
    commodities['bread'] = {
        'current': round(bread_now, 1),
        'past': round(bread_past, 1),
        'change_pct': round(bread_change, 1),
        'unit': 'loaves',
        'icon': '🍞'
    }
    
    # GASOLINE
    gas_now = current_value / CURRENT_PRICES['gas_gallon']
    gas_past = initial_investment / past_prices['gas_gallon']
    gas_change = ((gas_now - gas_past) / gas_past * 100) if gas_past > 0 else 0
    commodities['gas'] = {
        'current': round(gas_now, 1),
        'past': round(gas_past, 1),
        'change_pct': round(gas_change, 1),
        'unit': 'gallons',
        'icon': '⛽'
    }
    
    # WEIGHTED REAL PP INDEX (25% each)
    weights = {'eggs': 0.25, 'milk': 0.25, 'bread': 0.25, 'gas': 0.25}
    weighted_change = sum(commodities[k]['change_pct'] * weights[k] for k in weights)
    
    # Interpretation
    if weighted_change > 20:
        interpretation = "🟢 EXCELLENT - Significantly outpaced inflation"
    elif weighted_change > 5:
        interpretation = "🟢 GOOD - Preserved and grew purchasing power"
    elif weighted_change > -5:
        interpretation = "🟡 NEUTRAL - Roughly kept pace with inflation"
    elif weighted_change > -20:
        interpretation = "🟠 POOR - Lost purchasing power to inflation"
    else:
        interpretation = "🔴 VERY POOR - Severe loss of purchasing power"
    
    return {
        'commodities': commodities,
        'weighted_real_pp_change': round(weighted_change, 1),
        'interpretation': interpretation,
        'period_label': period_label,
        'pp_multiplier': round(pp_mult, 3),
        'initial_investment': initial_investment,
        'current_value': round(current_value, 2),
        # Legacy fields for compatibility
        'eggs_current_purchasing_power': commodities['eggs']['current'],
        'eggs_1y_ago_purchasing_power': commodities['eggs']['past'],
        'eggs_real_return_pct': commodities['eggs']['change_pct'],
        'eggs_interpretation': f"{'Gained' if commodities['eggs']['change_pct'] > 0 else 'Lost'} {abs(commodities['eggs']['change_pct']):.1f}%",
        'milk_current_purchasing_power': commodities['milk']['current'],
        'milk_1y_ago_purchasing_power': commodities['milk']['past'],
        'milk_real_return_pct': commodities['milk']['change_pct'],
        'milk_interpretation': f"{'Gained' if commodities['milk']['change_pct'] > 0 else 'Lost'} {abs(commodities['milk']['change_pct']):.1f}%"
    }

# =============================================================================
# PREDICTION ENGINE
# =============================================================================

def make_prediction(asset: str, horizon_years: int, model_type: str, models: Dict, test_data: pd.DataFrame) -> Optional[Dict]:
    """Make PPP-Q prediction for an asset - HORIZON AWARE"""
    
    if test_data is None or len(test_data) == 0:
        return None
    
    # Get latest data for asset
    asset_data = test_data[test_data['Asset'] == asset].sort_values('Date')
    if len(asset_data) == 0:
        return None
    
    latest_row = asset_data.iloc[-1]
    category = get_asset_category(asset)
    
    # Calculate component scores WITH HORIZON - THIS IS KEY!
    component_scores = calculate_component_scores(latest_row, asset, horizon_years)
    final_score = component_scores['final_composite_score']
    
    # Assign grade - with horizon adjustment
    # Longer horizons = more forgiving on volatility
    adjusted_score = final_score
    if horizon_years >= 7:
        # Long-term: boost score if growth potential is high
        growth_bonus = min(10, (latest_row.get('PP_Multiplier_10Y', 1.0) - 1.0) * 20)
        adjusted_score = min(100, final_score + growth_bonus)
    elif horizon_years <= 2:
        # Short-term: penalize high volatility more
        vol_penalty = min(10, latest_row.get('Volatility_90D', 0) / 10)
        adjusted_score = max(0, final_score - vol_penalty)
    
    grade = assign_grade(adjusted_score, category)
    grade_map = {'A': 'A_PRESERVER', 'B': 'B_PARTIAL', 'C': 'C_ERODER', 'D': 'D_DESTROYER'}
    predicted_class = grade_map.get(grade, 'C_ERODER')
    
    # Generate insights with horizon context
    strengths, weaknesses = generate_insights(latest_row, adjusted_score, grade)
    
    # Add horizon-specific insights
    if horizon_years >= 7:
        if latest_row.get('PP_Multiplier_10Y', 1.0) > 1.5:
            strengths.append(f"📈 Strong long-term growth: {((latest_row.get('PP_Multiplier_10Y', 1.0) - 1) * 100):.0f}% over 10Y")
    elif horizon_years <= 2:
        if latest_row.get('Volatility_90D', 0) < 20:
            strengths.append(f"🛡️ Low short-term volatility: {latest_row.get('Volatility_90D', 0):.1f}%")
        elif latest_row.get('Volatility_90D', 0) > 50:
            weaknesses.append(f"⚠️ High short-term volatility: {latest_row.get('Volatility_90D', 0):.1f}%")
    
    # Current status
    current_status = get_current_status(latest_row)
    
    # Real commodity comparison WITH HORIZON
    commodity_comparison = calculate_real_commodity_comparison(latest_row, asset, horizon_years)
    
    # Metrics adjusted by horizon
    if horizon_years <= 2:
        metrics = {
            'pp_multiplier': round(latest_row.get('PP_Multiplier_1Y', 1.0), 3),
            'sharpe_ratio': round(latest_row.get('Sharpe_Ratio_1Y', latest_row.get('Sharpe_Ratio_5Y', 0)), 3),
            'max_drawdown': round(latest_row.get('Max_Drawdown', 0), 1),
            'real_return': round(latest_row.get('Real_Return_1Y', latest_row.get('Real_Return_5Y', 0)), 1),
            'volatility_90d': round(latest_row.get('Volatility_90D', 0), 1),
            'horizon_label': f'{horizon_years}Y'
        }
    elif horizon_years <= 5:
        metrics = {
            'pp_multiplier': round(latest_row.get('PP_Multiplier_5Y', 1.0), 3),
            'sharpe_ratio': round(latest_row.get('Sharpe_Ratio_5Y', 0), 3),
            'max_drawdown': round(latest_row.get('Max_Drawdown', 0), 1),
            'real_return': round(latest_row.get('Real_Return_5Y', 0), 1),
            'volatility_90d': round(latest_row.get('Volatility_90D', 0), 1),
            'horizon_label': f'{horizon_years}Y'
        }
    else:
        metrics = {
            'pp_multiplier': round(latest_row.get('PP_Multiplier_10Y', latest_row.get('PP_Multiplier_5Y', 1.0)), 3),
            'sharpe_ratio': round(latest_row.get('Sharpe_Ratio_5Y', 0), 3),
            'max_drawdown': round(latest_row.get('Max_Drawdown', 0) * 0.7, 1),  # Long-term: drawdowns recover
            'real_return': round(latest_row.get('Real_Return_10Y', latest_row.get('Real_Return_5Y', 0)), 1),
            'volatility_90d': round(latest_row.get('Volatility_90D', 0), 1),
            'horizon_label': f'{horizon_years}Y'
        }
    
    # Confidence based on data quality, model agreement, and horizon
    base_confidence = 60 + adjusted_score * 0.35
    # Longer horizons = slightly lower confidence due to uncertainty
    horizon_penalty = min(10, (horizon_years - 3) * 1.5) if horizon_years > 3 else 0
    confidence = min(95, base_confidence - horizon_penalty)
    
    return {
        'asset': asset,
        'predicted_class': predicted_class,
        'confidence': round(confidence, 1),
        'component_scores': component_scores,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'current_status': current_status,
        'metrics': metrics,
        'real_commodity_comparison': commodity_comparison,
        'horizon_years': horizon_years,
        'model_type': model_type,
        'adjusted_score': round(adjusted_score, 1)
    }

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_gauge_chart(score: float, title: str) -> go.Figure:
    """Create gauge chart for PPP-Q score"""
    if score >= 65:
        color = "#28a745"
    elif score >= 55:
        color = "#17a2b8"
    elif score >= 42:
        color = "#ffc107"
    else:
        color = "#dc3545"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        number={'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 42], 'color': '#ffebee'},
                {'range': [42, 55], 'color': '#fff3e0'},
                {'range': [55, 65], 'color': '#e3f2fd'},
                {'range': [65, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': score}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_component_radar(components: Dict) -> go.Figure:
    """Create radar chart for component scores"""
    categories = ['Purchasing Power', 'Volatility Risk', 'Market Cycle', 
                  'Growth Potential', 'Consistency', 'Recovery', 'Risk-Adjusted']
    values = [
        components.get('real_purchasing_power_score', 0),
        components.get('volatility_risk_score', 0),
        components.get('market_cycle_score', 0),
        components.get('growth_potential_score', 0),
        components.get('consistency_score', 0),
        components.get('recovery_score', 0),
        components.get('risk_adjusted_score', 0)
    ]
    values.append(values[0])  # Close the polygon
    categories.append(categories[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        fillcolor='rgba(31, 119, 180, 0.3)',
        line=dict(color='#1f77b4', width=2),
        name='Component Scores'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, height=400, margin=dict(l=60, r=60, t=40, b=40)
    )
    return fig

def create_eggs_milk_comparison(commodity_data: Dict, asset: str) -> go.Figure:
    """Create comprehensive bar chart comparing purchasing power across all commodities"""
    
    # Check if we have the new commodity structure
    if 'commodities' in commodity_data:
        commodities = commodity_data['commodities']
        period_label = commodity_data.get('period_label', '1 Year Ago')
        weighted_change = commodity_data.get('weighted_real_pp_change', 0)
        
        # Create subplots for 4 commodities
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"{commodities['eggs']['icon']} Eggs (Dozen)",
                f"{commodities['milk']['icon']} Milk (Gallons)",
                f"{commodities['bread']['icon']} Bread (Loaves)",
                f"{commodities['gas']['icon']} Gasoline (Gallons)"
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        commodity_list = [
            ('eggs', 1, 1, '#FFD700', '#FFA500'),  # Gold/Orange
            ('milk', 1, 2, '#87CEEB', '#4169E1'),  # Sky blue/Royal blue
            ('bread', 2, 1, '#DEB887', '#8B4513'),  # Tan/Saddle brown
            ('gas', 2, 2, '#90EE90', '#228B22')    # Light green/Forest green
        ]
        
        for name, row, col, past_color, current_color in commodity_list:
            comm = commodities[name]
            x_labels = [period_label, 'Current']
            y_values = [comm['past'], comm['current']]
            change = comm['change_pct']
            
            # Color based on gain/loss
            bar_colors = [past_color, '#28a745' if change >= 0 else '#dc3545']
            
            fig.add_trace(go.Bar(
                x=x_labels, y=y_values,
                marker_color=bar_colors,
                text=[f"{v:.1f}" for v in y_values],
                textposition='outside',
                name=name.title(),
                hovertemplate=f"<b>{name.title()}</b><br>" +
                              "Period: %{x}<br>" +
                              f"{comm['unit'].title()}: " + "%{y:.1f}<br>" +
                              f"Change: {change:+.1f}%<extra></extra>"
            ), row=row, col=col)
            
            # Add change annotation
            fig.add_annotation(
                x=1, y=max(y_values) * 1.15,
                text=f"<b>{change:+.1f}%</b>",
                showarrow=False,
                font=dict(size=12, color='#28a745' if change >= 0 else '#dc3545'),
                row=row, col=col
            )
        
        # Overall title with weighted index
        title_color = '#28a745' if weighted_change >= 0 else '#dc3545'
        fig.update_layout(
            title=dict(
                text=f"<b>Real Purchasing Power: $100 in {asset}</b><br>" +
                     f"<span style='font-size:14px;color:{title_color}'>Weighted Real PP Change: {weighted_change:+.1f}%</span>",
                font=dict(size=16)
            ),
            height=500,
            showlegend=False,
            margin=dict(l=40, r=40, t=100, b=40)
        )
        
    else:
        # Legacy format - eggs and milk only
        fig = make_subplots(rows=1, cols=2, subplot_titles=('🥚 Eggs (Dozens)', '🥛 Milk (Gallons)'))
        
        # Eggs comparison
        eggs_data = ['1 Year Ago', 'Current']
        eggs_values = [
            commodity_data.get('eggs_1y_ago_purchasing_power', 25),
            commodity_data.get('eggs_current_purchasing_power', 25)
        ]
        eggs_colors = ['#90caf9', '#1976d2' if eggs_values[1] >= eggs_values[0] else '#e53935']
        
        fig.add_trace(go.Bar(x=eggs_data, y=eggs_values, marker_color=eggs_colors, 
                             text=[f"{v:.0f}" for v in eggs_values], textposition='outside',
                             name='Eggs'), row=1, col=1)
        
        # Milk comparison
        milk_values = [
            commodity_data.get('milk_1y_ago_purchasing_power', 25),
            commodity_data.get('milk_current_purchasing_power', 25)
        ]
        milk_colors = ['#a5d6a7', '#43a047' if milk_values[1] >= milk_values[0] else '#e53935']
        
        fig.add_trace(go.Bar(x=eggs_data, y=milk_values, marker_color=milk_colors,
                             text=[f"{v:.0f}" for v in milk_values], textposition='outside',
                             name='Milk'), row=1, col=2)
        
        fig.update_layout(
            title=f"Real Purchasing Power: $100 in {asset}",
            height=350, showlegend=False,
            margin=dict(l=40, r=40, t=80, b=40)
        )
    
    return fig

def create_multi_asset_comparison(results: List[Dict]) -> go.Figure:
    """Create comprehensive multi-asset comparison chart"""
    if not results:
        return None
    
    assets = [r['asset'] for r in results]
    scores = [r['component_scores']['final_composite_score'] for r in results]
    grades = [r['predicted_class'].split('_')[0] for r in results]
    
    colors = ['#28a745' if g == 'A' else '#17a2b8' if g == 'B' else '#ffc107' if g == 'C' else '#dc3545' for g in grades]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=assets, y=scores, marker_color=colors,
        text=[f"{s:.1f}<br>({g})" for s, g in zip(scores, grades)],
        textposition='outside'
    ))
    
    # Add grade threshold lines
    fig.add_hline(y=65, line_dash="dash", line_color="green", annotation_text="A Grade")
    fig.add_hline(y=55, line_dash="dash", line_color="blue", annotation_text="B Grade")
    fig.add_hline(y=42, line_dash="dash", line_color="orange", annotation_text="C Grade")
    
    fig.update_layout(
        title="PPP-Q Score Comparison",
        xaxis_title="Asset", yaxis_title="PPP-Q Score",
        yaxis=dict(range=[0, max(scores) + 15] if scores else [0, 100]),
        height=450
    )
    return fig

def create_correlation_heatmap(test_data: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap based on actual return time series"""
    # Get unique assets
    assets = test_data['Asset'].unique().tolist()[:8]  # Limit to 8 for readability
    
    # Build a pivot table of returns over time for each asset
    # We need time-series data to calculate meaningful correlations
    
    # Check for Date column
    if 'Date' not in test_data.columns:
        # Fallback: create simple comparison matrix
        return create_simple_comparison_matrix(test_data, assets)
    
    # Try to get return column - use PP_Multiplier_1Y or Real_Return_1Y
    return_col = None
    for col in ['Real_Return_1Y', 'PP_Multiplier_1Y', 'Real_Return_5Y', 'PP_Multiplier_5Y']:
        if col in test_data.columns:
            return_col = col
            break
    
    if return_col is None:
        return create_simple_comparison_matrix(test_data, assets)
    
    # Create pivot table: Date x Asset with returns
    try:
        pivot_df = test_data.pivot_table(
            index='Date', 
            columns='Asset', 
            values=return_col, 
            aggfunc='mean'
        )
        
        # Only keep assets with enough data points
        pivot_df = pivot_df[assets].dropna(axis=1, thresh=len(pivot_df) // 2)
        
        if len(pivot_df.columns) < 2:
            return create_simple_comparison_matrix(test_data, assets)
        
        # Calculate correlation matrix
        corr_matrix = pivot_df.corr()
        
    except Exception:
        return create_simple_comparison_matrix(test_data, assets)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale='RdBu', zmid=0,
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Asset Return Correlation Matrix (based on {return_col})", 
        height=450
    )
    return fig

def create_simple_comparison_matrix(test_data: pd.DataFrame, assets: list) -> go.Figure:
    """Create a comparison matrix based on PPP-Q component scores"""
    # Get latest data for each asset and compute diverse metrics
    asset_metrics = []
    
    for asset in assets:
        asset_df = test_data[test_data['Asset'] == asset]
        if len(asset_df) == 0:
            continue
            
        latest = asset_df.iloc[-1]
        
        # Get multiple metrics to create variance
        metrics = {
            'asset': asset,
            'pp_mult_5y': latest.get('PP_Multiplier_5Y', 1.0),
            'volatility': latest.get('Volatility_90D', 50),
            'sharpe': latest.get('Sharpe_Ratio_5Y', 0),
            'max_dd': abs(latest.get('Max_Drawdown', 0)),
            'real_return': latest.get('Real_Return_5Y', 0)
        }
        asset_metrics.append(metrics)
    
    if len(asset_metrics) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for correlation", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create DataFrame and calculate correlation between assets based on their metrics profile
    df = pd.DataFrame(asset_metrics).set_index('asset')
    
    # Calculate similarity between assets (correlation of their metric profiles)
    # Transpose so we correlate assets against each other
    corr_matrix = df.T.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale='RdBu', zmid=0,
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        hovertemplate='%{x} vs %{y}<br>Similarity: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(title="Asset Similarity Matrix (based on metric profiles)", height=450)
    return fig

# =============================================================================
# PART 2: REAL DATA PIPELINE & RETRAINING
# =============================================================================

def fetch_live_data_from_apis() -> pd.DataFrame:
    """Fetch live economic and asset data from public APIs and return as DataFrame"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_data = []
    
    # Comprehensive asset ticker mapping (ALL assets)
    tickers = {
        # Cryptocurrencies
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD',
        'Litecoin': 'LTC-USD',
        'Ripple': 'XRP-USD',
        'Cardano': 'ADA-USD',
        'Solana': 'SOL-USD',
        'Dogecoin': 'DOGE-USD',
        'Polkadot': 'DOT-USD',
        'Avalanche': 'AVAX-USD',
        'Chainlink': 'LINK-USD',
        'Polygon': 'MATIC-USD',
        'Uniswap': 'UNI-USD',
        
        # Precious Metals (Futures)
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Platinum': 'PL=F',
        'Palladium': 'PA=F',
        'Copper': 'HG=F',
        
        # Major Indices
        'SP500': '^GSPC',
        'NASDAQ': '^IXIC',
        'DowJones': '^DJI',
        'Russell2000': '^RUT',
        'VIX': '^VIX',
        'FTSE100': '^FTSE',
        'DAX': '^GDAXI',
        'Nikkei225': '^N225',
        'HangSeng': '^HSI',
        
        # Large Cap Tech Stocks
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Google': 'GOOGL',
        'Amazon': 'AMZN',
        'Meta': 'META',
        'Tesla': 'TSLA',
        'Nvidia': 'NVDA',
        'Netflix': 'NFLX',
        'Adobe': 'ADBE',
        'Salesforce': 'CRM',
        'Intel': 'INTC',
        'AMD': 'AMD',
        'Qualcomm': 'QCOM',
        'Oracle': 'ORCL',
        'IBM': 'IBM',
        
        # Financial Stocks
        'JPMorgan': 'JPM',
        'BankOfAmerica': 'BAC',
        'WellsFargo': 'WFC',
        'GoldmanSachs': 'GS',
        'MorganStanley': 'MS',
        'Citigroup': 'C',
        'Visa': 'V',
        'Mastercard': 'MA',
        'PayPal': 'PYPL',
        'AmericanExpress': 'AXP',
        'Berkshire': 'BRK-B',
        
        # Healthcare
        'JohnsonJohnson': 'JNJ',
        'UnitedHealth': 'UNH',
        'Pfizer': 'PFE',
        'Merck': 'MRK',
        'AbbVie': 'ABBV',
        'Eli Lilly': 'LLY',
        'Moderna': 'MRNA',
        
        # Consumer/Retail
        'Walmart': 'WMT',
        'Costco': 'COST',
        'HomeDepot': 'HD',
        'McDonalds': 'MCD',
        'Nike': 'NKE',
        'Starbucks': 'SBUX',
        'CocaCola': 'KO',
        'PepsiCo': 'PEP',
        'Procter&Gamble': 'PG',
        
        # Energy
        'Oil': 'CL=F',
        'NaturalGas': 'NG=F',
        'ExxonMobil': 'XOM',
        'Chevron': 'CVX',
        'Shell': 'SHEL',
        'BP': 'BP',
        'ConocoPhillips': 'COP',
        
        # ETFs - Diversified
        'Gold_ETF': 'GLD',
        'Silver_ETF': 'SLV',
        'SP500_ETF': 'SPY',
        'Nasdaq_ETF': 'QQQ',
        'TreasuryBond_ETF': 'TLT',
        'RealEstate_ETF': 'VNQ',
        'EmergingMarkets_ETF': 'VWO',
        'International_ETF': 'VXUS',
        'SmallCap_ETF': 'IWM',
        'Value_ETF': 'VTV',
        'Growth_ETF': 'VUG',
        'Dividend_ETF': 'VYM',
        'HighYieldBond_ETF': 'HYG',
        'CorpBond_ETF': 'LQD',
        'Bitcoin_ETF': 'IBIT',
        'Ethereum_ETF': 'ETHE',
        
        # REITs
        'AmericanTower': 'AMT',
        'Prologis': 'PLD',
        'CrownCastle': 'CCI',
        'Equinix': 'EQIX',
        'SimonProperty': 'SPG',
        
        # Industrial/Materials
        'Caterpillar': 'CAT',
        'Deere': 'DE',
        'Boeing': 'BA',
        'Honeywell': 'HON',
        '3M': 'MMM',
        'UPS': 'UPS',
        
        # Telecommunications
        'Verizon': 'VZ',
        'AT&T': 'T',
        'T-Mobile': 'TMUS',
    }
    
    try:
        import yfinance as yf
        
        for idx, (name, ticker) in enumerate(tickers.items()):
            progress_bar.progress((idx + 1) / len(tickers))
            status_text.text(f"Fetching {name}...")
            
            try:
                data = yf.Ticker(ticker)
                # Get 5 years of data for proper calculations
                hist = data.history(period="5y")
                
                if len(hist) < 252:  # Need at least 1 year
                    continue
                
                # Calculate metrics for latest data point
                latest_date = hist.index[-1].strftime('%Y-%m-%d')
                latest_price = hist['Close'].iloc[-1]
                
                # Returns
                return_1y = (hist['Close'].iloc[-1] / hist['Close'].iloc[-252] - 1) if len(hist) >= 252 else 0
                return_5y = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) if len(hist) >= 252 else return_1y
                
                # Volatility
                daily_returns = hist['Close'].pct_change().dropna()
                volatility_90d = daily_returns.tail(90).std() * np.sqrt(252) * 100
                volatility_1y = daily_returns.tail(252).std() * np.sqrt(252) * 100
                
                # Max Drawdown
                rolling_max = hist['Close'].cummax()
                drawdown = (hist['Close'] - rolling_max) / rolling_max
                max_drawdown = drawdown.min() * 100
                
                # Sharpe Ratio (assuming 4% risk-free rate)
                risk_free = 0.04
                excess_return_1y = return_1y - risk_free
                sharpe_1y = (excess_return_1y / (volatility_1y / 100)) if volatility_1y > 0 else 0
                sharpe_5y = ((return_5y / 5) - risk_free) / (volatility_1y / 100) if volatility_1y > 0 else 0
                
                # PP Multipliers (inflation-adjusted - assuming 3% annual inflation)
                inflation_1y = 0.03
                inflation_5y = (1.03 ** 5) - 1
                pp_mult_1y = (1 + return_1y) / (1 + inflation_1y)
                pp_mult_5y = (1 + return_5y) / (1 + inflation_5y)
                pp_mult_10y = pp_mult_5y ** 2  # Extrapolated
                
                # Real Returns
                real_return_1y = (pp_mult_1y - 1) * 100
                real_return_5y = (pp_mult_5y - 1) * 100
                
                # Recovery Score (how quickly it recovers from drawdowns)
                recovery_periods = []
                in_drawdown = False
                dd_start = 0
                for i, dd in enumerate(drawdown):
                    if dd < -0.05 and not in_drawdown:
                        in_drawdown = True
                        dd_start = i
                    elif dd >= -0.01 and in_drawdown:
                        recovery_periods.append(i - dd_start)
                        in_drawdown = False
                avg_recovery = np.mean(recovery_periods) if recovery_periods else 90
                
                # Create data row
                row = {
                    'Date': latest_date,
                    'Asset': name,
                    f'{name}_Price': latest_price,
                    'Return_1Y': return_1y * 100,
                    'Return_5Y': return_5y * 100,
                    'Volatility_90D': volatility_90d,
                    'Volatility_1Y': volatility_1y,
                    'Max_Drawdown': max_drawdown,
                    'Sharpe_Ratio_1Y': sharpe_1y,
                    'Sharpe_Ratio_5Y': sharpe_5y,
                    'PP_Multiplier_1Y': pp_mult_1y,
                    'PP_Multiplier_5Y': pp_mult_5y,
                    'PP_Multiplier_10Y': pp_mult_10y,
                    'Real_Return_1Y': real_return_1y,
                    'Real_Return_5Y': real_return_5y,
                    'Avg_Recovery_Days': avg_recovery,
                    'Data_Source': 'yfinance_live',
                    'Fetch_Timestamp': datetime.now().isoformat()
                }
                
                all_data.append(row)
                
            except Exception as e:
                st.warning(f"Could not fetch {name}: {str(e)[:50]}")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Data fetch complete!")
        
    except ImportError:
        st.error("❌ yfinance not installed. Run: pip install yfinance")
        return pd.DataFrame()
    
    if all_data:
        return pd.DataFrame(all_data)
    return pd.DataFrame()


def preprocess_new_data(live_df: pd.DataFrame, existing_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess live data and merge with existing dataset"""
    
    if live_df.empty:
        return existing_data
    
    st.info("⚙️ Running preprocessing pipeline...")
    
    # Ensure all required columns exist
    required_cols = ['Date', 'Asset', 'PP_Multiplier_1Y', 'PP_Multiplier_5Y', 'PP_Multiplier_10Y',
                     'Volatility_90D', 'Sharpe_Ratio_5Y', 'Max_Drawdown', 'Real_Return_5Y']
    
    # Fill missing columns with defaults
    for col in required_cols:
        if col not in live_df.columns:
            if 'Multiplier' in col:
                live_df[col] = 1.0
            elif 'Volatility' in col:
                live_df[col] = 30.0
            elif 'Sharpe' in col:
                live_df[col] = 0.0
            elif 'Return' in col or 'Drawdown' in col:
                live_df[col] = 0.0
    
    # Calculate additional PPP-Q features
    for idx, row in live_df.iterrows():
        # Inflation Protection Score
        pp_mult = row.get('PP_Multiplier_5Y', 1.0)
        live_df.at[idx, 'Inflation_Protection_Score'] = min(100, max(0, (pp_mult - 0.5) * 100))
        
        # Volatility Score (lower is better)
        vol = row.get('Volatility_90D', 50)
        live_df.at[idx, 'Volatility_Score'] = max(0, 100 - vol)
        
        # Risk-Adjusted Score
        sharpe = row.get('Sharpe_Ratio_5Y', 0)
        live_df.at[idx, 'Risk_Adjusted_Score'] = min(100, max(0, (sharpe + 1) * 33))
        
        # Composite PPP-Q Score
        live_df.at[idx, 'Composite_Score'] = (
            live_df.at[idx, 'Inflation_Protection_Score'] * 0.4 +
            live_df.at[idx, 'Volatility_Score'] * 0.3 +
            live_df.at[idx, 'Risk_Adjusted_Score'] * 0.3
        )
    
    # Merge with existing data
    if existing_data is not None and len(existing_data) > 0:
        # Add new data (avoid duplicates by Date+Asset)
        if 'Date' in existing_data.columns and 'Asset' in existing_data.columns:
            existing_keys = set(zip(existing_data['Date'], existing_data['Asset']))
            new_rows = live_df[~live_df.apply(lambda r: (r['Date'], r['Asset']) in existing_keys, axis=1)]
            
            if len(new_rows) > 0:
                combined = pd.concat([existing_data, new_rows], ignore_index=True)
                st.success(f"✅ Added {len(new_rows)} new records to dataset")
                return combined
            else:
                st.info("ℹ️ No new unique records to add (data already exists)")
                return existing_data
        else:
            return pd.concat([existing_data, live_df], ignore_index=True)
    
    return live_df


def save_data_locally(df: pd.DataFrame, filename: str = "pppq_updated.csv"):
    """Save processed data to local file"""
    try:
        base = get_base_path()
        output_path = os.path.join(base, "..", "data", "processed", "pppq", filename)
        output_path = os.path.normpath(output_path)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        return output_path
    except Exception as e:
        st.error(f"Error saving: {e}")
        return None


def retrain_models_with_new_data(train_data: pd.DataFrame):
    """Actually retrain the ML models with new data"""
    
    if train_data is None or len(train_data) < 100:
        st.error("❌ Need at least 100 samples to retrain")
        return False
    
    progress = st.progress(0)
    status = st.empty()
    
    try:
        # Prepare features
        status.text("Preparing features...")
        progress.progress(10)
        
        # Define target (PPP-Q Grade)
        def assign_grade_label(row):
            score = row.get('Composite_Score', 50)
            if score >= 70:
                return 'A_PRESERVER'
            elif score >= 55:
                return 'B_PARTIAL'
            elif score >= 35:
                return 'C_ERODER'
            else:
                return 'D_DESTROYER'
        
        train_data['Target'] = train_data.apply(assign_grade_label, axis=1)
        
        # Feature columns
        feature_cols = [col for col in train_data.columns if col not in 
                       ['Date', 'Asset', 'Target', 'Data_Source', 'Fetch_Timestamp'] 
                       and train_data[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        X = train_data[feature_cols].fillna(0)
        y = train_data['Target']
        
        # Encode target
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        status.text("Training LightGBM...")
        progress.progress(30)
        
        # Train LightGBM
        import lightgbm as lgb
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        lgb_params = {
            'objective': 'multiclass',
            'num_class': len(le.classes_),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1
        }
        
        lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=100, 
                              valid_sets=[lgb_val], callbacks=[lgb.early_stopping(10)])
        
        status.text("Training XGBoost...")
        progress.progress(60)
        
        # Train XGBoost
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            objective='multi:softprob',
            num_class=len(le.classes_),
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        status.text("Evaluating models...")
        progress.progress(80)
        
        # Evaluate
        from sklearn.metrics import accuracy_score, f1_score
        
        lgb_pred = lgb_model.predict(X_val).argmax(axis=1)
        xgb_pred = xgb_model.predict(X_val)
        
        lgb_acc = accuracy_score(y_val, lgb_pred)
        xgb_acc = accuracy_score(y_val, xgb_pred)
        lgb_f1 = f1_score(y_val, lgb_pred, average='weighted')
        xgb_f1 = f1_score(y_val, xgb_pred, average='weighted')
        
        status.text("Saving models...")
        progress.progress(90)
        
        # Save models
        base = get_base_path()
        models_dir = os.path.normpath(os.path.join(base, "..", "models", "pppq"))
        os.makedirs(models_dir, exist_ok=True)
        
        # Save LightGBM
        lgb_model.save_model(os.path.join(models_dir, "lgbm_model.txt"))
        
        # Save XGBoost
        xgb_model.save_model(os.path.join(models_dir, "xgb_model.json"))
        
        # Save feature columns
        with open(os.path.join(models_dir, "feature_columns.json"), 'w') as f:
            json.dump(feature_cols, f)
        
        progress.progress(100)
        status.text("✅ Training complete!")
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("LightGBM Accuracy", f"{lgb_acc*100:.1f}%")
            st.metric("LightGBM F1", f"{lgb_f1*100:.1f}%")
        with col2:
            st.metric("XGBoost Accuracy", f"{xgb_acc*100:.1f}%")
            st.metric("XGBoost F1", f"{xgb_f1*100:.1f}%")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Training error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False


def run_data_pipeline():
    """Run the complete data preprocessing and retraining pipeline"""
    st.subheader("🔧 Data Pipeline & Model Retraining")
    
    # Initialize session state
    if 'live_data' not in st.session_state:
        st.session_state['live_data'] = None
    if 'preprocessed_data' not in st.session_state:
        st.session_state['preprocessed_data'] = None
    
    # Step 1: Data Collection
    st.markdown("### Step 1: 📥 Fetch Live Data")
    st.caption("Fetch real-time data from Yahoo Finance for assets")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🔄 Fetch Live Data", type="primary", key="fetch_data_btn"):
            with st.spinner("Connecting to APIs..."):
                live_df = fetch_live_data_from_apis()
                if not live_df.empty:
                    st.session_state['live_data'] = live_df
                    st.success(f"✅ Fetched {len(live_df)} asset records")
    
    with col2:
        if st.session_state['live_data'] is not None:
            st.dataframe(st.session_state['live_data'][['Asset', 'Date', 'PP_Multiplier_5Y', 'Volatility_90D', 'Sharpe_Ratio_5Y']].round(3), 
                        use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Step 2: Preprocessing
    st.markdown("### Step 2: ⚙️ Preprocess & Add to Dataset")
    st.caption("Calculate PPP-Q features and merge with existing data")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("⚙️ Preprocess Data", key="preprocess_btn"):
            if st.session_state['live_data'] is not None:
                # Get existing data from the loaded models
                models = initialize_models()
                existing = models.get('test_data')
                
                preprocessed = preprocess_new_data(st.session_state['live_data'], existing)
                st.session_state['preprocessed_data'] = preprocessed
                
                # Save to file
                saved_path = save_data_locally(preprocessed, "pppq_updated.csv")
                if saved_path:
                    st.success(f"✅ Saved to: {saved_path}")
            else:
                st.warning("⚠️ Fetch data first!")
    
    with col2:
        if st.session_state['preprocessed_data'] is not None:
            st.metric("Total Records", f"{len(st.session_state['preprocessed_data']):,}")
            if 'Asset' in st.session_state['preprocessed_data'].columns:
                st.metric("Unique Assets", st.session_state['preprocessed_data']['Asset'].nunique())
    
    st.markdown("---")
    
    # Step 3: Model Retraining
    st.markdown("### Step 3: 🤖 Retrain Models")
    st.caption("Train new LightGBM and XGBoost models with updated data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Retraining will:**
        - Train new LightGBM model
        - Train new XGBoost model  
        - Evaluate on validation set
        - Save models to disk
        """)
    
    with col2:
        if st.button("🚀 Retrain Models", type="primary", key="retrain_btn"):
            data_to_train = st.session_state.get('preprocessed_data')
            if data_to_train is None:
                # Use existing data if no new data
                models = initialize_models()
                data_to_train = models.get('test_data')
            
            if data_to_train is not None and len(data_to_train) > 0:
                success = retrain_models_with_new_data(data_to_train)
                if success:
                    st.balloons()
                    st.success("🎉 Models retrained successfully! Restart app to load new models.")
                    # Clear cache to reload models
                    st.cache_resource.clear()
            else:
                st.error("❌ No data available for training")
    
    # Download option
    st.markdown("---")
    st.markdown("### 📥 Export Data")
    if st.session_state['preprocessed_data'] is not None:
        csv = st.session_state['preprocessed_data'].to_csv(index=False)
        st.download_button(
            label="📥 Download Updated Dataset",
            data=csv,
            file_name=f"pppq_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 PPP-Q Investment Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Purchasing Power Preservation Quotient - Self-Contained ML Dashboard</p>', unsafe_allow_html=True)
    
    # Initialize models and data
    with st.spinner("Loading models and data from GitHub..."):
        models = initialize_models()
        test_data = models.get('test_data')
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Data Source Status
        st.subheader("📡 Data Source")
        if test_data is not None and len(test_data) > 0:
            st.success(f"✅ Loaded {len(test_data)} records")
            if 'Asset' in test_data.columns:
                st.caption(f"Assets: {test_data['Asset'].nunique()}")
            else:
                st.caption(f"Columns: {len(test_data.columns)}")
                st.warning(f"Available columns: {list(test_data.columns)[:5]}...")
        else:
            st.error("❌ Data not loaded")
            st.caption("Check GitHub URLs")
        
        st.markdown("---")
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        model_type = st.selectbox(
            "Select Model",
            options=["ensemble", "lgbm", "xgb"],
            format_func=lambda x: MODEL_INFO[x]["name"]
        )
        st.info(f"**Accuracy:** {MODEL_INFO[model_type]['accuracy']}")
        st.caption(MODEL_INFO[model_type]['description'])
        
        st.markdown("---")
        
        # Horizon Selection
        st.subheader("📅 Investment Horizon")
        horizon_years = st.slider("Years", min_value=1, max_value=10, value=5)
        
        if horizon_years < 2:
            st.caption("**Short-term:** Focus on liquidity")
        elif horizon_years <= 5:
            st.caption("**Medium-term:** Balanced approach")
        else:
            st.caption("**Long-term:** Growth focus")
    
    # Main Content Tabs
    tabs = st.tabs([
        "🎯 Single Asset",
        "📊 Compare Assets", 
        "📈 Correlations",
        "🔧 Data Pipeline",
        "📚 Documentation"
    ])
    
    # ==========================================================================
    # Tab 1: Single Asset Analysis
    # ==========================================================================
    with tabs[0]:
        st.header("🎯 Single Asset Analysis")
        
        if test_data is None or len(test_data) == 0:
            st.error("Data not loaded. Check GitHub connection.")
            return
        
        if 'Asset' not in test_data.columns:
            st.error(f"Missing 'Asset' column. Available columns: {list(test_data.columns)[:10]}")
            return
        
        available_assets = sorted(test_data['Asset'].unique().tolist())
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_asset = st.selectbox("Select Asset", options=available_assets)
        with col2:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        # Asset info
        if selected_asset:
            asset_info = ASSETS.get(selected_asset, {"category": "Unknown", "symbol": "N/A", "risk": "Unknown", "liquidity": "Unknown"})
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Category", asset_info.get("category", "Unknown"))
            col2.metric("Symbol", asset_info.get("symbol", "N/A"))
            col3.metric("Risk Level", asset_info.get("risk", "Unknown"))
            col4.metric("Liquidity", asset_info.get("liquidity", "Unknown"))
        
        if analyze_btn and selected_asset:
            with st.spinner("Analyzing..."):
                result = make_prediction(selected_asset, horizon_years, model_type, models, test_data)
            
            if result:
                st.markdown("---")
                
                # Main Score Display
                grade = result['predicted_class'].split('_')[0]
                final_score = result['component_scores']['final_composite_score']
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.plotly_chart(create_gauge_chart(final_score, "PPP-Q Score"), use_container_width=True)
                
                with col2:
                    grade_colors = {'A': 'grade-A', 'B': 'grade-B', 'C': 'grade-C', 'D': 'grade-D'}
                    grade_desc = {'A': 'Excellent - Strong PP preservation', 'B': 'Good - Above average', 
                                  'C': 'Fair - Moderate protection', 'D': 'Poor - High risk'}
                    
                    st.markdown(f"""
                    <div class="metric-card {grade_colors.get(grade, '')}">
                        <h1 style="font-size: 4rem; margin: 0;">Grade {grade}</h1>
                        <p style="font-size: 1.2rem;">{grade_desc.get(grade, '')}</p>
                        <p style="font-size: 0.9rem; opacity: 0.8;">{horizon_years}Y Horizon | {MODEL_INFO[model_type]['name']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.metric("Confidence", f"{result['confidence']:.1f}%")
                    st.metric("PP Multiplier", f"{result['metrics'].get('pp_multiplier', result['metrics'].get('pp_multiplier_5y', 1.0)):.2f}x")
                    st.metric("Sharpe Ratio", f"{result['metrics'].get('sharpe_ratio', result['metrics'].get('sharpe_ratio_5y', 0)):.2f}")
                
                st.markdown("---")
                
                # Component Breakdown
                st.subheader("📊 Component Score Breakdown")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.plotly_chart(create_component_radar(result['component_scores']), use_container_width=True)
                
                with col2:
                    components = result['component_scores']
                    for name, key in [
                        ('💰 Purchasing Power', 'real_purchasing_power_score'),
                        ('📉 Volatility Risk', 'volatility_risk_score'),
                        ('🔄 Market Cycle', 'market_cycle_score'),
                        ('🚀 Growth Potential', 'growth_potential_score'),
                        ('🎯 Consistency', 'consistency_score'),
                        ('💪 Recovery', 'recovery_score'),
                        ('⚖️ Risk-Adjusted', 'risk_adjusted_score')
                    ]:
                        score = components.get(key, 0)
                        st.progress(score / 100, text=f"{name}: {score:.1f}/100")
                
                st.markdown("---")
                
                # Real Purchasing Power Comparison (Full Commodity Basket)
                st.subheader("🛒 Real Purchasing Power Analysis")
                
                # Display interpretation summary
                commodity_data = result['real_commodity_comparison']
                if 'interpretation' in commodity_data:
                    interp = commodity_data['interpretation']
                    weighted_change = commodity_data.get('weighted_real_pp_change', 0)
                    current_value = commodity_data.get('current_value', 100)
                    period_label = commodity_data.get('period_label', 'Previously')
                    
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, {'#d4edda' if weighted_change >= 0 else '#f8d7da'}, white); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                        <h3 style="margin: 0 0 10px 0; color: #333;">{interp}</h3>
                        <p style="margin: 5px 0; color: #555; font-size: 1.1rem;">
                            <strong>$100 invested</strong> → <strong>${current_value:.2f}</strong> (based on {horizon_years}Y {period_label} comparison)
                        </p>
                        <p style="margin: 5px 0; color: #555;">
                            <strong>Weighted Basket Change:</strong> 
                            <span style="color: {'#28a745' if weighted_change >= 0 else '#dc3545'}; font-size: 1.2rem; font-weight: bold;">
                                {weighted_change:+.1f}%
                            </span>
                            <span style="font-size: 0.9rem; opacity: 0.8;"> (25% eggs, 25% milk, 25% bread, 25% gas)</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.plotly_chart(
                    create_eggs_milk_comparison(result['real_commodity_comparison'], selected_asset),
                    use_container_width=True
                )
                
                st.markdown("---")
                
                # Strengths & Weaknesses - FIXED VISIBILITY
                st.subheader("💡 AI-Generated Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### ✅ Strengths")
                    for s in result['strengths']:
                        st.markdown(f'<div class="success-box">{s}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### ⚠️ Weaknesses")
                    for w in result['weaknesses']:
                        st.markdown(f'<div class="danger-box">{w}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Current Market Status
                st.subheader("📈 Current Market Status")
                status = result['current_status']
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Volatility", status['volatility'], f"{status['volatility_value']}%")
                col2.metric("Cycle Position", status['cycle_position'])
                col3.metric("Distance from ATH", status['distance_from_ath'])
                col4.metric("Entry Signal", status['entry_signal'])
    
    # ==========================================================================
    # Tab 2: Compare Assets
    # ==========================================================================
    with tabs[1]:
        st.header("📊 Compare Multiple Assets")
        
        if test_data is None or 'Asset' not in test_data.columns:
            st.error("Data not loaded or missing Asset column")
            return
        
        available_assets = sorted(test_data['Asset'].unique().tolist())
        selected_assets = st.multiselect(
            "Select Assets (2-6)",
            options=available_assets,
            default=["Bitcoin", "Gold", "SP500"][:min(3, len(available_assets))]
        )
        
        if len(selected_assets) >= 2:
            if st.button("🔄 Compare", type="primary"):
                with st.spinner("Comparing..."):
                    results = []
                    for asset in selected_assets:
                        result = make_prediction(asset, horizon_years, model_type, models, test_data)
                        if result:
                            results.append(result)
                
                if results:
                    # Main comparison chart
                    st.plotly_chart(create_multi_asset_comparison(results), use_container_width=True)
                    
                    # Detailed comparison table
                    st.subheader("📋 Detailed Comparison")
                    table_data = []
                    for r in results:
                        metrics = r['metrics']
                        table_data.append({
                            "Asset": r['asset'],
                            "Grade": r['predicted_class'].split('_')[0],
                            "Score": r['component_scores']['final_composite_score'],
                            "Confidence": f"{r['confidence']:.1f}%",
                            "PP Mult": f"{metrics.get('pp_multiplier', metrics.get('pp_multiplier_5y', 1.0)):.2f}x",
                            "Sharpe": f"{metrics.get('sharpe_ratio', metrics.get('sharpe_ratio_5y', 0)):.2f}",
                            "Volatility": f"{metrics.get('volatility_90d', 0):.1f}%",
                            "Entry": r['current_status']['entry_signal']
                        })
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
                    
                    # Best pick
                    best = max(results, key=lambda x: x['component_scores']['final_composite_score'])
                    st.success(f"🏆 **Best Pick:** {best['asset']} (Score: {best['component_scores']['final_composite_score']:.1f})")
                    
                    # Component comparison
                    st.subheader("📊 Component Score Comparison")
                    comp_data = []
                    for r in results:
                        for comp, val in r['component_scores'].items():
                            if comp != 'final_composite_score':
                                comp_data.append({'Asset': r['asset'], 'Component': comp.replace('_score', '').replace('_', ' ').title(), 'Score': val})
                    
                    if comp_data:
                        fig = px.bar(pd.DataFrame(comp_data), x='Component', y='Score', color='Asset', barmode='group', height=400)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Select at least 2 assets")
    
    # ==========================================================================
    # Tab 3: Correlations & Insights
    # ==========================================================================
    with tabs[2]:
        st.header("📈 Correlations & Market Insights")
        
        if test_data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Asset Correlation Matrix")
                st.plotly_chart(create_correlation_heatmap(test_data), use_container_width=True)
            
            with col2:
                st.subheader("Asset Categories")
                categories = {}
                for asset, info in ASSETS.items():
                    cat = info.get('category', 'Other')
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append(asset)
                
                for cat, assets in categories.items():
                    with st.expander(f"**{cat}** ({len(assets)})"):
                        for asset in assets:
                            info = ASSETS[asset]
                            st.markdown(f"- **{asset}** ({info['symbol']}) - Risk: {info['risk']}")
        
        st.markdown("---")
        
        # Market Regime Indicators - FIXED VISIBILITY
        st.subheader("📊 Market Regime Indicators")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="regime-card">
                <h4>🏦 Inflation Environment</h4>
                <p>Current CPI trends affect asset valuations differently.</p>
                <p><strong>High Inflation Favors:</strong> Gold, Commodities, TIPS</p>
                <p><strong>Low Inflation Favors:</strong> Growth Stocks, Long Bonds</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="regime-card">
                <h4>📈 Growth vs Value</h4>
                <p>Economic cycle phase affects optimal allocation.</p>
                <p><strong>Expansion:</strong> Equities, Crypto outperform</p>
                <p><strong>Contraction:</strong> Bonds, Gold provide safety</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="regime-card">
                <h4>⚡ Volatility Regime</h4>
                <p>VIX levels indicate market stress.</p>
                <p><strong>Low VIX (&lt;15):</strong> Risk-on environment</p>
                <p><strong>High VIX (&gt;25):</strong> Defensive positioning</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ==========================================================================
    # Tab 4: Data Pipeline
    # ==========================================================================
    with tabs[3]:
        st.header("🔧 Data Pipeline & Model Retraining")
        run_data_pipeline()
        
        st.markdown("---")
        st.subheader("📊 Current Dataset Statistics")
        
        if test_data is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", f"{len(test_data):,}")
            col2.metric("Unique Assets", test_data['Asset'].nunique() if 'Asset' in test_data.columns else 'N/A')
            col3.metric("Features", f"{len(test_data.columns)}")
            col4.metric("Date Range", f"{test_data['Date'].min()[:10]} → {test_data['Date'].max()[:10]}" if 'Date' in test_data else "N/A")
            
            # Full Data View with Filters
            st.subheader("📋 Full Dataset Explorer")
            
            # Filters
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                show_all = st.checkbox("Show All Data", value=False, help="Toggle to see full dataset")
            
            with filter_col2:
                if 'Asset' in test_data.columns:
                    filter_asset = st.selectbox(
                        "Filter by Asset",
                        options=["All Assets"] + sorted(test_data['Asset'].unique().tolist()),
                        key="data_asset_filter"
                    )
                else:
                    filter_asset = "All Assets"
                    st.warning("No 'Asset' column found")
            
            with filter_col3:
                num_rows = st.slider(
                    "Rows to Display",
                    min_value=10,
                    max_value=min(500, len(test_data)),
                    value=50 if not show_all else len(test_data),
                    step=10,
                    disabled=show_all,
                    key="data_rows_slider"
                )
            
            # Apply filters
            display_data = test_data.copy()
            if filter_asset != "All Assets" and 'Asset' in display_data.columns:
                display_data = display_data[display_data['Asset'] == filter_asset]
            
            # Sort by date descending (most recent first)
            if 'Date' in display_data.columns:
                display_data = display_data.sort_values('Date', ascending=False)
            
            # Display count
            if show_all:
                rows_to_show = len(display_data)
            else:
                rows_to_show = min(num_rows, len(display_data))
            
            st.info(f"📊 Showing {rows_to_show:,} of {len(display_data):,} records" + 
                   (f" for {filter_asset}" if filter_asset != "All Assets" else ""))
            
            # Scrollable dataframe with all data
            st.dataframe(
                display_data.head(rows_to_show) if not show_all else display_data,
                use_container_width=True,
                height=400
            )
            
            # Download option
            st.download_button(
                label="📥 Download Full Dataset as CSV",
                data=display_data.to_csv(index=False),
                file_name="pppq_full_dataset.csv",
                mime="text/csv"
            )
            
            # Asset breakdown
            if 'Asset' in test_data.columns:
                st.subheader("📈 Asset Distribution")
                asset_counts = test_data['Asset'].value_counts()
                st.bar_chart(asset_counts)
    
    # ==========================================================================
    # Tab 5: Documentation
    # ==========================================================================
    with tabs[4]:
        st.header("📚 Documentation")
        
        st.markdown("""
        ## PPP-Q Model Overview
        
        The **Purchasing Power Preservation Quotient (PPP-Q)** is a machine learning system 
        that evaluates assets based on their ability to preserve purchasing power over time.
        
        ### Grading System
        
        | Grade | Score Range | Interpretation |
        |-------|-------------|----------------|
        | **A** | ≥ 65-75* | Excellent - Strong purchasing power preservation |
        | **B** | 55-64 | Good - Above average protection |
        | **C** | 35-54 | Fair - Moderate protection with risks |
        | **D** | < 35 | Poor - High risk to purchasing power |
        
        *Crypto assets require higher scores (≥75) due to volatility
        
        ### The 7 Component Scores
        
        | Component | Weight | Description |
        |-----------|--------|-------------|
        | **Real Purchasing Power** | 25% | Can you buy MORE goods with this asset vs cash and other commidities? |
        | **Volatility Risk** | 20% | How stable is this asset? (Higher = more stable) |
        | **Market Cycle** | 15% | Is now a good time to buy? |
        | **Growth Potential** | 15% | How much MORE can this grow? |
        | **Consistency** | 10% | Reliable returns or boom-bust? |
        | **Recovery** | 10% | Bounces back fast from crashes? |
        | **Risk-Adjusted** | 5% | Quality of returns per unit of risk |
        
        ### Data Sources
        
        - **GitHub Repository:** Raw data and trained models
        - **Yahoo Finance:** Real-time price data (via yfinance)
        - **FRED API:** Economic indicators (CPI, Interest Rates)
        
        ### Self-Contained Architecture
        
        This dashboard runs **100% on Streamlit Cloud**:
        - No external API server required
        - Models loaded directly from GitHub
        - Data fetched from GitHub raw URLs
        - Optional live data from Yahoo Finance
        
        ---
        
        **Version:** 2.0.0 (Self-Contained)  
        **Author:** Bilal Ahmad Sheikh  
        **Last Updated:** December 2025
        """)

# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()
