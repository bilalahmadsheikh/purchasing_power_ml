# 📊 Incremental ML Pipeline Documentation

## Overview

The **PPP-Q Incremental ML Pipeline** is an automated workflow that runs every 15 days to:
- Fetch ONLY new economic and market data (not re-fetch everything)
- Append new rows to the existing consolidated dataset
- Preprocess only the new data rows
- Retrain models on the complete dataset (existing + new)
- Automatically deploy if performance improves
- Send email notifications to stakeholders

This approach is **resource-efficient** and **time-saving** compared to full retrains that re-fetch 10+ years of historical data.

---

## Architecture

### Components

```
src/pipelines/
├── pipeline_config.py      # Central configuration (paths, params, assets)
├── notifications.py        # Email notifications to ba8616127@gmail.com
├── model_registry.py       # MLflow model versioning and tracking
├── prefect_flows.py        # Main Prefect orchestration (5 tasks)
└── __init__.py
```

### Task Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INCREMENTAL PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TASK 1: FETCH NEW DATA (Incremental)                          │
│  ├─ Check last_date in final_consolidated_dataset.csv          │
│  ├─ Fetch ONLY data from APIs where Date > last_date            │
│  ├─ Feature engineer new rows                                   │
│  ├─ Append to existing consolidated CSV                         │
│  └─ Return: (complete_data, new_rows_count, new_data_only)     │
│                           ↓                                      │
│  TASK 2: PREPROCESS DATA (Incremental)                         │
│  ├─ Process only NEW rows                                       │
│  ├─ Feature extraction for PPPQ classification                  │
│  ├─ Append to train/val/test splits based on date ranges        │
│  └─ Return: (train_df, val_df, test_df)                         │
│                           ↓                                      │
│  TASK 3: TRAIN MODELS (on All Data)                            │
│  ├─ Train LightGBM                                              │
│  ├─ Train XGBoost                                               │
│  ├─ Train Random Forest                                         │
│  ├─ Create ensemble (weighted voting)                           │
│  └─ Return: Training metrics for all models                     │
│                           ↓                                      │
│  TASK 4: EVALUATE & VERSION (MLflow)                           │
│  ├─ Evaluate on test set                                        │
│  ├─ Compare with previous best model                            │
│  ├─ Register with MLflow if better                              │
│  ├─ Decide deployment based on thresholds                       │
│  └─ Return: Evaluation metrics, deploy decision                 │
│                           ↓                                      │
│  TASK 5: SEND NOTIFICATIONS (Email)                            │
│  ├─ Notify pipeline start                                       │
│  ├─ Notify success/failure                                      │
│  ├─ Include metrics and new data stats                          │
│  └─ Send to: ba8616127@gmail.com                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### 1️⃣ Data Ingestion (INCREMENTAL)

**File:** `src/pipelines/prefect_flows.py` → `fetch_new_data()`

#### Logic Flow:
```python
# 1. Check existing data
if final_consolidated_dataset.csv exists:
    last_date = max(Date) in CSV
    existing_rows = count of rows
else:
    last_date = None (first run)

# 2. Fetch fresh data from all sources
df_economic = fetch_economic_data()           # FRED API
df_assets = fetch_asset_and_vix_prices()      # Yahoo Finance
df_crypto = fetch_crypto_data_yfinance()      # Yahoo Finance
df_commodities = fetch_real_baselines()       # FRED API
df_global = fetch_global_market_data()        # World Bank API

# 3. Merge all sources
df_merged = merge_all_raw_data(...)

# 4. FILTER TO ONLY NEW DATA
if last_date is not None:
    df_new_only = df_merged[df_merged['Date'] > last_date]
    new_rows = len(df_new_only)
else:
    df_new_only = df_merged  # first run
    new_rows = len(df_new_only)

# 5. Feature engineering on NEW data
df_new_featured = engineer_features(df_new_only)

# 6. Append to existing CSV
if existing_data:
    df_combined = concat([existing_df, df_new_featured])
    df_combined.drop_duplicates(subset=['Date'])
    df_combined.to_csv(final_consolidated_dataset.csv)
else:
    df_new_featured.to_csv(final_consolidated_dataset.csv)

# 7. Return complete data + new data
return df_combined, new_rows, df_new_featured
```

#### Data Sources:
| Source | Data Type | API | Update Frequency |
|--------|-----------|-----|------------------|
| FRED | Economic indicators (inflation, GDP, unemployment) | https://fred.stlouisfed.org | Daily |
| Yahoo Finance | Stock prices, crypto prices, VIX | yfinance library | Daily |
| World Bank | Global M2, Global GDP | World Bank API | Monthly |
| CoinGecko | Crypto supply data | Free API | Daily |

#### New Data Detection:
- **First Run:** Fetches all historical data
- **Subsequent Runs:** Only fetches data where `Date > last_date_in_csv`
- **No New Data:** Pipeline skips (no wasted compute)

---

### 2️⃣ Preprocessing (INCREMENTAL)

**File:** `src/pipelines/prefect_flows.py` → `preprocess_data()`

#### What Gets Processed:
```python
# Input: Complete dataset + new_rows_count + df_new_only
def preprocess_data(df_raw, new_rows_count, df_new_only):
    
    # If no new data: return existing splits
    if new_rows_count == 0:
        return (train_df, val_df, test_df)  # load from disk
    
    # Otherwise: process ALL data but focus on NEW rows
    
    for each asset in CORE_ASSETS:
        # Create asset-specific row for each date
        # Extract features:
        # - Real returns (3Y, 5Y, 10Y)
        # - PP multipliers (1Y, 5Y, 10Y)  
        # - Volatility (90D)
        # - Sharpe ratio (5Y)
        # - Max drawdown
        # - Distance from ATH
        # - Distance from MA200
        # - Market cap saturation
        # - Composite score
        
        # Calculate PPP_Q_Composite_Score (weighted)
        # Assign PPP_Q_Class: A_PRESERVER, B_PARTIAL, C_ERODER, D_DESTROYER
    
    # Time-based splits (NO LEAKAGE)
    train_df = data where TRAIN_START <= Date <= TRAIN_END
    val_df   = data where VAL_START <= Date <= VAL_END
    test_df  = data where TEST_START <= Date <= TEST_END
    
    # Save to CSV
    train_df.to_csv(data/processed/pppq/train/)
    val_df.to_csv(data/processed/pppq/val/)
    test_df.to_csv(data/processed/pppq/test/)
    
    return (train_df, val_df, test_df)
```

#### Asset Categories:
```python
CORE_ASSETS = {
    'Bitcoin': crypto,
    'Ethereum': crypto,
    'Gold': precious_metal,
    'Silver': precious_metal,
    'WTI_Crude': commodity,
    'Natural_Gas': commodity,
    'S&P_500': equity_index,
    'Nasdaq_100': equity_index,
    'AAPL': tech_stock,
    'MSFT': tech_stock,
    'NVDA': tech_stock,
    'TESLA': tech_stock,
}
```

#### PPP_Q Classes:
| Class | Score Range | Meaning |
|-------|-------------|---------|
| **A_PRESERVER** | 65-100+ | Excellent inflation protection |
| **B_PARTIAL** | 45-65 | Moderate inflation protection |
| **C_ERODER** | 25-45 | Weak inflation protection |
| **D_DESTROYER** | 0-25 | No protection / value loss |

---

### 3️⃣ Model Training

**File:** `src/pipelines/prefect_flows.py` → `train_models()`

#### Models Trained:
1. **LightGBM** (Gradient Boosting)
   - Params: max_depth=7, learning_rate=0.05, n_estimators=500
   - Best for: Fast training, good interpretability

2. **XGBoost** (Extreme Gradient Boosting)
   - Params: max_depth=6, learning_rate=0.1, n_estimators=500
   - Best for: Regularization, reduced overfitting

3. **Random Forest**
   - Params: n_estimators=200, max_depth=10
   - Best for: Feature importance, robustness

4. **Ensemble** (Weighted Voting)
   - Weights: LightGBM=0.4, XGBoost=0.35, RandomForest=0.25
   - Combines strengths of all models

#### Training Output:
```
models/pppq/
├── lgbm_model.txt                 # LightGBM model (text format)
├── xgb_model.json                 # XGBoost model (JSON format)
├── feature_columns.json           # Feature names used
├── model_registry.json            # MLflow tracking
└── training_summary.json          # Metrics for all models
```

---

### 4️⃣ Evaluation & Deployment

**File:** `src/pipelines/prefect_flows.py` → `evaluate_and_version()`

#### Metrics Tracked:
```python
# Classification metrics for PPP_Q_Class prediction
metrics = {
    'accuracy': overall correctness,
    'macro_f1': average F1 across all classes,
    'balanced_accuracy': average recall across classes,
    'precision_per_class': {A, B, C, D},
    'recall_per_class': {A, B, C, D},
    'f1_per_class': {A, B, C, D},
    'confusion_matrix': [[TP, FP], [FN, TN]],
    'roc_auc_macro': area under ROC curve
}
```

#### Deployment Decision Logic:
```python
# Auto-deploy if:
1. New model Macro F1 > previous best + 0.01  (1% improvement)
2. Accuracy > 0.65 (65% minimum)
3. Balanced Accuracy > 0.60 (60% minimum)

# MLflow Tracking:
- Register model with version number
- Tag with: new_data_count, deployment_date, metrics
- Store in: models/pppq/model_registry.json

# Production Ready:
- Deployed models saved to: models/pppq/production/
- API served from: src/api/main.py
- Endpoint: POST /api/v1/predict
```

---

### 5️⃣ Notifications

**File:** `src/pipelines/notifications.py`

#### Notification Types:

**📧 Pipeline Start**
```
Subject: 🚀 PPP-Q Pipeline Started
To: ba8616127@gmail.com
Content: Pipeline execution started at [timestamp]
```

**📧 Pipeline Success**
```
Subject: ✅ PPP-Q Pipeline Completed Successfully
To: ba8616127@gmail.com

Details:
- New Data Rows: 150
- Best Model: LightGBM
- Macro F1: 0.7823
- Accuracy: 0.8124
- Deployed: ✅ Yes
- Run ID: pppq-v42
```

**📧 Pipeline Failure**
```
Subject: ❌ PPP-Q Pipeline Failed
To: ba8616127@gmail.com

Error: [error message]
Time: [timestamp]
```

**📧 Model Deployed**
```
Subject: 🚀 New Model Deployed to Production
To: ba8616127@gmail.com

Model Version: pppq-v42
Macro F1: 0.7823
Accuracy: 0.8124
```

---

## Running the Pipeline

### Manual Execution

**Incremental Update (default)**
```bash
python run_pipeline.py
```
Fetches only new data since last run.

**Force Full Retrain**
```bash
python run_pipeline.py --force
```
Retrains even if no new data available.

**Scheduled Execution (every 15 days)**
```bash
python run_pipeline.py --schedule
```
Uses APScheduler to run on schedule.

**Test Email Configuration**
```bash
python run_pipeline.py --test-email
```
Sends test email to verify .env setup.

### Automated Execution

**Via GitHub Actions** (every 15 days)
- Workflow: `.github/workflows/automated-pipeline.yml`
- Schedule: `0 0 1,15 * *` (1st and 15th of each month)
- Runs on: `ubuntu-latest`
- Steps:
  1. Checkout code
  2. Set up Python 3.11
  3. Install dependencies
  4. Run `python run_pipeline.py`
  5. Commits updated data/models (if changed)
  6. Pushes to repository

---

## Configuration

### Environment Variables (.env)

```bash
# API Keys
FRED_API_KEY=your_fred_api_key
YAHOO_FINANCE_API_KEY=optional

# Email Notification
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
RECIPIENT_EMAIL=ba8616127@gmail.com

# Prefect Cloud (optional)
PREFECT_API_KEY=your_prefect_api_key
PREFECT_API_URL=https://api.prefect.cloud/api

# MLflow
MLFLOW_TRACKING_URI=file:./mlruns
```

### Data Paths (pipeline_config.py)

```python
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'final_consolidated_dataset.csv'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'pppq'
TRAIN_DATA = PROCESSED_DIR / 'train' / 'pppq_train.csv'
VAL_DATA = PROCESSED_DIR / 'val' / 'pppq_val.csv'
TEST_DATA = PROCESSED_DIR / 'test' / 'pppq_test.csv'
MODELS_DIR = PROJECT_ROOT / 'models' / 'pppq'
```

### Time-Based Splits (pipeline_config.py)

```python
TRAIN_START = '2015-01-01'
TRAIN_END = '2022-12-31'     # ~70% of data
VAL_START = '2023-01-01'
VAL_END = '2023-06-30'       # ~15% of data
TEST_START = '2023-07-01'
TEST_END = '2023-12-31'      # ~15% of data
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  External Data Sources                                              │
├─────────────────────────────────────────────────────────────────────┤
│  FRED API         Yahoo Finance      World Bank      CoinGecko      │
│  └─Economic        └─Prices          └─Global M2     └─Supply       │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TASK 1: FETCH NEW DATA                                             │
│  └─ Filters to dates > last_date                                    │
│  └─ Features engineering                                            │
│  └─ Appends to final_consolidated_dataset.csv                       │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  data/raw/final_consolidated_dataset.csv                            │
│  (Complete historical + new rows)                                   │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TASK 2: PREPROCESS DATA                                            │
│  └─ Asset-level feature extraction                                  │
│  └─ PPP_Q classification scores                                     │
│  └─ Time-based splits (train/val/test)                              │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  data/processed/pppq/                                               │
│  ├─ train/pppq_train.csv    (2015-2022)                            │
│  ├─ val/pppq_val.csv        (2023-06)                              │
│  └─ test/pppq_test.csv      (2023-12)                              │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TASK 3: TRAIN MODELS                                               │
│  └─ LightGBM, XGBoost, Random Forest, Ensemble                      │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  models/pppq/                                                       │
│  ├─ lgbm_model.txt                                                  │
│  ├─ xgb_model.json                                                  │
│  └─ model_registry.json (MLflow tracking)                           │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TASK 4: EVALUATE & VERSION                                         │
│  └─ Compare with previous best                                      │
│  └─ Register with MLflow if better                                  │
│  └─ Decide auto-deployment                                          │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TASK 5: SEND NOTIFICATIONS                                         │
│  └─ Email to ba8616127@gmail.com                                    │
│  └─ Metrics, new data stats, deployment info                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Monitoring & Troubleshooting

### Pipeline Logs
```bash
# View recent pipeline logs
cat logs/pipeline_*.log

# Follow real-time logs (if running)
tail -f logs/pipeline_$(date +%Y%m%d).log
```

### Check Data Freshness
```bash
# Last date in consolidated dataset
python -c "
import pandas as pd
df = pd.read_csv('data/raw/final_consolidated_dataset.csv')
print(f'Total rows: {len(df)}')
print(f'Date range: {df[\"Date\"].min()} to {df[\"Date\"].max()}')
"
```

### Check Model Performance
```bash
# View latest metrics
cat models/pppq/training_summary.json | python -m json.tool
```

### Common Issues

| Issue | Solution |
|-------|----------|
| **No new data detected** | Check FRED/Yahoo APIs are accessible. Date parsing issue? |
| **Preprocessing fails** | Check data types in final_consolidated_dataset.csv. Missing columns? |
| **Model training slow** | Reduce n_estimators in pipeline_config.py. Data too large? |
| **Email not sending** | Verify SMTP credentials in .env. Gmail needs app password. |
| **GitHub Actions fails** | Check secrets are set. FRED_API_KEY in repo settings? |

---

## Performance Metrics

### Typical Pipeline Runtime
- **Data Ingestion:** 2-5 minutes (depends on API response times)
- **Preprocessing:** 1-2 minutes (70 assets × dates)
- **Model Training:** 3-8 minutes (3 models + ensemble)
- **Evaluation:** 30 seconds
- **Notifications:** 5 seconds
- **Total:** ~7-17 minutes

### Model Performance (Baseline)
- **LightGBM Macro F1:** 0.78
- **XGBoost Macro F1:** 0.76
- **Ensemble Macro F1:** 0.79
- **Accuracy:** 81%

---

## Files Modified by Pipeline

Each run updates:
```
data/raw/
  └─ final_consolidated_dataset.csv          ← NEW ROWS APPENDED

data/processed/pppq/
  ├─ train/pppq_train.csv                    ← Updated
  ├─ val/pppq_val.csv                        ← Updated
  ├─ test/pppq_test.csv                      ← Updated
  ├─ pppq_features.json                      ← Feature list
  ├─ pppq_summary.json                       ← Dataset stats
  └─ pppq_thresholds.json                    ← Class thresholds

models/pppq/
  ├─ lgbm_model.txt                          ← New version
  ├─ xgb_model.json                          ← New version
  ├─ feature_columns.json                    ← Feature names
  ├─ model_registry.json                     ← MLflow history
  └─ training_summary.json                   ← Metrics

logs/
  └─ pipeline_YYYYMMDD_HHMMSS.log           ← Execution log
```

---

## Next Steps & Improvements

- [ ] Add data quality checks (missing values, outliers)
- [ ] Implement automated feature selection
- [ ] Add hyperparameter tuning with Optuna
- [ ] Implement model explainability (SHAP values)
- [ ] Add performance alerts (F1 drops below threshold)
- [ ] Scale to production with Kubernetes
- [ ] Add more data sources (news sentiment, on-chain metrics)

---

**Last Updated:** December 2024  
**Maintainer:** Bilal Ahmad Sheikh  
**Pipeline Type:** Incremental | **Frequency:** Every 15 days  
**Notification:** Email to ba8616127@gmail.com
