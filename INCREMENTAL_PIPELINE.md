# 📊 Incremental ML Pipeline Documentation

## Overview (v2.0.0)

The **PPP-Q Incremental ML Pipeline** is an automated workflow that runs every 15 days to:
- Fetch ONLY new economic and market data (not re-fetch everything)
- Append new rows to the existing consolidated dataset
- Preprocess data with **39 features** including egg/milk commodity basket
- **Train 10 multi-output models** (2 classifiers + 8 component regressors)
- **Apply horizon-aware predictions** (dynamic adjustments for 1Y-10Y investments)
- Automatically deploy if performance improves
- Send email notifications to stakeholders

This approach is **resource-efficient** and **time-saving** compared to full retrains that re-fetch 10+ years of historical data.

---

## v2.0.0 Major Enhancements

### 🚀 Multi-Output ML Architecture

**Classification Stage** (Ensemble):
- **LightGBM Classifier:** 96.5% Macro F1 (40% weight)
- **XGBoost Classifier:** 96.7% Macro F1 (60% weight)
- **Ensemble:** 96.30% Macro F1 on test set

**Regression Stage** (8 Component Scores):
- **8 LightGBM Regressors:** 99.3% avg R² on test set
- Each predicts a component score (0-100)
- Weighted composite determines final class

### 📊 Component Scores (NEW in v2.0.0)

| Component | Weight | What It Measures | Model | R² |
|-----------|--------|------------------|-------|-----|
| **Real Purchasing Power** | 25% | PP preservation vs inflation | LightGBM | 99.5% |
| **Volatility Risk** | 20% | Price stability (inverse volatility) | LightGBM | 99.2% |
| **Market Cycle** | 15% | Buy-low opportunity (distance from ATH) | LightGBM | 98.8% |
| **Growth Potential** | 15% | Future appreciation (market cap saturation) | LightGBM | 99.1% |
| **Consistency** | 10% | Return reliability over time | LightGBM | 98.5% |
| **Recovery** | 10% | Bounce-back speed from crashes | LightGBM | 98.2% |
| **Risk-Adjusted** | 15% | Returns per unit of risk (Sharpe-like) | LightGBM | 99.0% |
| **Commodity Score** | 5% | PP vs eggs & milk basket (NEW!) | LightGBM | 99.4% |

### 🎯 Horizon-Aware Predictions

**Feature Adjustments** based on investment timeframe:

| Adjustment Type | 1-Year Horizon | 5-Year Horizon | 10-Year Horizon |
|----------------|----------------|----------------|-----------------|
| **Volatility Weight** | High (25%) | Medium (20%) | Low (15%) - time diversifies |
| **Growth Weight** | Low (10%) | Medium (15%) | High (20%) - compounding |
| **Cycle Weight** | High (20%) | Medium (15%) | Low (10%) - less relevant |
| **PP Multiplier** | 1Y used | 5Y used | 10Y used |

**Example**: Bitcoin volatile short-term → C_ERODER at 1Y, but A_PRESERVER at 10Y

### 🥚 Commodity Features (NEW)

5 new features tracking real-world purchasing power:

1. **Eggs_Per_100USD**: How many dozen eggs $100 can buy
2. **Milk_Gallons_Per_100USD**: How many gallons of milk $100 can buy
3. **Real_Return_Eggs_1Y**: 1-year return vs egg price inflation
4. **Real_Return_Milk_1Y**: 1-year return vs milk price inflation
5. **Real_Commodity_Basket_Return_1Y**: Blended egg/milk performance

**Why**: Eggs and milk are universal consumer staples that everyone buys. Better indicator of real purchasing power than abstract inflation metrics.

---

## Architecture

### Components

```
src/pipelines/
├── pipeline_config.py      # Central configuration (paths, params, assets)
├── notifications.py        # Email notifications to ba8616127@gmail.com
├── model_registry.py       # MLflow model versioning + MODEL_ARTIFACTS_V2
├── prefect_flows.py        # Main Prefect orchestration (v2.0.0 updated)
└── __init__.py

src/models/
├── pppq_multi_output_model.py  # v2.0.0 training script (10 models)
└── train_lgb_xgb.py            # DEPRECATED (v1.x)
```

### Task Flow (v2.0.0)

```
┌─────────────────────────────────────────────────────────────────┐
│            INCREMENTAL PIPELINE v2.0.0                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TASK 1: FETCH NEW DATA (Incremental)                          │
│  ├─ Check last_date in final_consolidated_dataset.csv          │
│  ├─ Fetch ONLY data from APIs where Date > last_date            │
│  ├─ Feature engineer new rows (39 features with commodities)    │
│  ├─ Append to existing consolidated CSV                         │
│  └─ Return: (complete_data, new_rows_count, new_data_only)     │
│                           ↓                                      │
│  TASK 2: PREPROCESS DATA (Incremental)                         │
│  ├─ Process only NEW rows                                       │
│  ├─ Feature extraction for PPPQ (39 features total)             │
│  ├─ Calculate 8 component score targets                         │
│  ├─ Append to train/val/test splits based on date ranges        │
│  └─ Return: (train_df, val_df, test_df)                         │
│                           ↓                                      │
│  TASK 3: TRAIN MULTI-OUTPUT MODELS (v2.0.0)                    │
│  ├─ Train 2 Classifiers (LightGBM + XGBoost)                   │
│  ├─ Train 8 Component Regressors (LightGBM)                    │
│  ├─ Validate all 10 models created                             │
│  ├─ Load metrics from training_metrics_v2.json                 │
│  └─ Return: Classification F1 + Component R² metrics            │
│                           ↓                                      │
│  TASK 4: EVALUATE & VERSION (MLflow)                           │
│  ├─ Evaluate classification ensemble (96.3% F1)                │
│  ├─ Evaluate component regressors (99.3% avg R²)               │
│  ├─ Compare with previous best model                            │
│  ├─ Register with MLflow (v2.0.0 metadata)                     │
│  ├─ Decide deployment based on F1 threshold                    │
│  └─ Return: Evaluation metrics, deploy decision                 │
│                           ↓                                      │
│  TASK 5: SEND NOTIFICATIONS (Email)                            │
│  ├─ Notify pipeline start                                       │
│  ├─ Notify success/failure                                      │
│  ├─ Include v2.0.0 metrics (10 models, Classification F1,      │
│  │   Component Avg R², Commodity Score)                        │
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

# NEW in v2.0.0: Commodity prices (eggs & milk)
df_commodity_prices = fetch_commodity_prices()  # BLS API or manual

# 3. Merge all sources
df_merged = merge_all_raw_data(...)

# 4. FILTER TO ONLY NEW DATA
if last_date is not None:
    df_new_only = df_merged[df_merged['Date'] > last_date]
    new_rows = len(df_new_only)
else:
    df_new_only = df_merged  # first run
    new_rows = len(df_new_only)

# 5. Feature engineering on NEW data (39 features)
df_new_featured = engineer_features(df_new_only)

# NEW in v2.0.0: Add commodity features
df_new_featured = add_commodity_features(df_new_featured, df_commodity_prices)

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
| **BLS** (NEW) | **Egg & milk prices** | Manual or BLS API | Monthly |

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
        # Extract features (39 total):
        # - Real returns (3Y, 5Y, 10Y)
        # - PP multipliers (1Y, 5Y, 10Y)
        # - Volatility (90D)
        # - Sharpe ratio (5Y)
        # - Max drawdown
        # - Distance from ATH
        # - Distance from MA200
        # - Market cap saturation
        # - Composite score
        # - Stability & consistency metrics
        # - Commodity features (eggs, milk) ← NEW!

        # NEW in v2.0.0: Calculate 8 component score targets
        targets = {
            'real_pp_score': calculate_real_pp_score(row),
            'volatility_score': calculate_volatility_score(row),
            'cycle_score': calculate_cycle_score(row),
            'growth_score': calculate_growth_score(row),
            'consistency_score': calculate_consistency_score(row),
            'recovery_score': calculate_recovery_score(row),
            'risk_adjusted_score': calculate_risk_adjusted_score(row),
            'commodity_score': calculate_commodity_score(row)  # NEW!
        }

        # Calculate PPP_Q_Composite_Score (weighted from components)
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

#### PPP_Q Classes (Dynamic Thresholds):
| Class | Base Score | Crypto Threshold | Metal Threshold | Index Threshold |
|-------|------------|------------------|-----------------|-----------------|
| **A_PRESERVER** | ≥65 | ≥70 | ≥62 | ≥65 |
| **B_PARTIAL** | 55-64 | 58-69 | 52-61 | 55-64 |
| **C_ERODER** | 42-54 | 45-57 | 40-51 | 42-54 |
| **D_DESTROYER** | <42 | <45 | <40 | <42 |

*Note: Thresholds adjust based on asset category and investment horizon*

---

### 3️⃣ Model Training (v2.0.0 - Multi-Output)

**File:** `src/models/pppq_multi_output_model.py` (called by prefect_flows.py)

#### Models Trained (10 Total):

**Classification Models (2)**:
1. **LightGBM Classifier**
   - Params: `num_leaves=31, learning_rate=0.05, num_iterations=500`
   - Performance: 96.5% Macro F1, best iteration=186
   - Weight in ensemble: 40%

2. **XGBoost Classifier**
   - Params: `max_depth=7, learning_rate=0.05, n_estimators=500`
   - Performance: 96.7% Macro F1
   - Weight in ensemble: 60%

**Component Regression Models (8 LightGBM Regressors)**:
3. Real Purchasing Power Score (99.5% R²)
4. Volatility Risk Score (99.2% R²)
5. Market Cycle Score (98.8% R²)
6. Growth Potential Score (99.1% R²)
7. Consistency Score (98.5% R²)
8. Recovery Score (98.2% R²)
9. Risk-Adjusted Score (99.0% R²)
10. **Commodity Score (99.4% R²)** ← NEW in v2.0.0!

#### Training Output:
```
models/pppq/
├── lgbm_classifier.txt                              # Classification ensemble (40%)
├── xgb_classifier.json                              # Classification ensemble (60%)
├── lgbm_target_real_pp_score_regressor.txt         # Component 1
├── lgbm_target_volatility_score_regressor.txt      # Component 2
├── lgbm_target_cycle_score_regressor.txt           # Component 3
├── lgbm_target_growth_score_regressor.txt          # Component 4
├── lgbm_target_consistency_score_regressor.txt     # Component 5
├── lgbm_target_recovery_score_regressor.txt        # Component 6
├── lgbm_target_risk_adjusted_score_regressor.txt   # Component 7
├── lgbm_target_commodity_score_regressor.txt       # Component 8 (NEW!)
├── feature_columns.json                             # 39 features
├── label_encoder.pkl                                # Class encoder
├── model_registry.json                              # MLflow tracking
└── training_metrics_v2.json                         # v2.0.0 metrics
```

---

### 4️⃣ Evaluation & Deployment

**File:** `src/pipelines/prefect_flows.py` → `evaluate_and_version()`

#### Metrics Tracked (v2.0.0):
```python
# Classification metrics
classification_metrics = {
    'macro_f1': 0.963,              # Primary deployment metric
    'accuracy': 0.965,
    'balanced_accuracy': 0.962,
    'precision_per_class': {...},
    'recall_per_class': {...},
    'f1_per_class': {...}
}

# Component regression metrics (NEW)
component_metrics = {
    'avg_r2': 0.993,                # Average across 8 regressors
    'min_r2': 0.982,                # Worst performer (Recovery)
    'max_r2': 0.995,                # Best performer (Real PP)
    'rmse_avg': 1.2,
    'individual_scores': {
        'real_pp_score_r2': 0.995,
        'volatility_score_r2': 0.992,
        'cycle_score_r2': 0.988,
        'growth_score_r2': 0.991,
        'consistency_score_r2': 0.985,
        'recovery_score_r2': 0.982,
        'risk_adjusted_score_r2': 0.990,
        'commodity_score_r2': 0.994   # NEW!
    }
}
```

#### Deployment Decision Logic:
```python
# Auto-deploy if:
1. New classification F1 > previous best + 0.001  (0.1% improvement)
2. Classification F1 > 0.90 (90% minimum)
3. Component avg R² > 0.95 (95% minimum)

# MLflow Tracking (v2.0.0):
- Register model with version number
- Tag with:
  * version: '2.0.0'
  * num_models: 10
  * classifiers: 2
  * regressors: 8
  * classification_f1: 0.963
  * component_avg_r2: 0.993
  * new_data_count: X
  * deployment_date: timestamp

# Production Ready:
- Deployed models saved to: models/pppq/
- API served from: src/api/main.py (predict_ml.py)
- Endpoint: POST /predict
```

---

### 5️⃣ Notifications (v2.0.0 Updated)

**File:** `src/pipelines/notifications.py`

#### Notification Types:

**📧 Pipeline Start**
```
Subject: 🚀 PPP-Q Pipeline v2.0.0 Started
To: ba8616127@gmail.com
Content: Pipeline execution started at [timestamp]
```

**📧 Pipeline Success (v2.0.0)**
```
Subject: ✅ PPP-Q Pipeline v2.0.0 Completed Successfully
To: ba8616127@gmail.com

Details:
- Model Version: v2.0.0 (Multi-Output)
- Models Trained: 10 (2 classifiers + 8 regressors)
- New Data Rows: 150
- Classification F1: 0.9630 (96.30%)
- Classification Accuracy: 0.9650
- Component Avg R²: 0.9930 (99.30%)
- Commodity Score R²: 0.9940 (NEW!)
- Deployed: ✅ Yes
- Run ID: pppq-v2-20241217
```

**📧 Pipeline Failure**
```
Subject: ❌ PPP-Q Pipeline v2.0.0 Failed
To: ba8616127@gmail.com

Error: [error message]
Time: [timestamp]
Failed Task: [task name]
```

**📧 Model Deployed (v2.0.0)**
```
Subject: 🚀 New Multi-Output Model Deployed to Production
To: ba8616127@gmail.com

Model Version: pppq-v2-20241217
Architecture: Multi-Output (2 classifiers + 8 regressors)
Classification F1: 0.9630
Component Avg R²: 0.9930
Models Deployed: 10
```

---

## Running the Pipeline

### Manual Execution

**Incremental Update (default)**
```bash
python -c "from src.pipelines.prefect_flows import run_pipeline; run_pipeline()"
```
Fetches only new data since last run, trains all 10 models.

**Force Full Retrain**
```bash
python -c "from src.pipelines.prefect_flows import run_pipeline; run_pipeline(force_full_retrain=True)"
```
Retrains even if no new data available.

**Scheduled Execution**
```bash
python -c "from src.pipelines.prefect_flows import schedule_pipeline; schedule_pipeline()"
```
Runs every 15 days automatically.

### Automated Execution

**Via GitHub Actions** (every 15 days)
- Workflow: `.github/workflows/automated-pipeline.yml`
- Schedule: `0 2 */15 * *` (every 15 days at 2 AM UTC)
- Runs on: `ubuntu-latest`
- Steps:
  1. Checkout code with Git LFS
  2. Set up Python 3.10
  3. Install dependencies
  4. Run data collection
  5. Run preprocessing (v2.0.0 with 39 features)
  6. Train multi-output models (10 models)
  7. Validate all models exist
  8. Commits updated data/models (Git LFS)
  9. Pushes to repository
  10. Send notification

---

## Configuration

### Environment Variables (.env)

```bash
# API Keys
FRED_API_KEY=your_fred_api_key
YAHOO_FINANCE_API_KEY=optional
BLS_API_KEY=optional  # NEW in v2.0.0 for egg/milk prices

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

### Model Registry (v2.0.0)

```python
# NEW in v2.0.0: Tracks all 10 models
MODEL_ARTIFACTS_V2 = {
    # Classifiers
    'lgbm_classifier': 'models/pppq/lgbm_classifier.txt',
    'xgb_classifier': 'models/pppq/xgb_classifier.json',

    # Component Regressors
    'lgbm_real_pp': 'models/pppq/lgbm_target_real_pp_score_regressor.txt',
    'lgbm_volatility': 'models/pppq/lgbm_target_volatility_score_regressor.txt',
    'lgbm_cycle': 'models/pppq/lgbm_target_cycle_score_regressor.txt',
    'lgbm_growth': 'models/pppq/lgbm_target_growth_score_regressor.txt',
    'lgbm_consistency': 'models/pppq/lgbm_target_consistency_score_regressor.txt',
    'lgbm_recovery': 'models/pppq/lgbm_target_recovery_score_regressor.txt',
    'lgbm_risk_adjusted': 'models/pppq/lgbm_target_risk_adjusted_score_regressor.txt',
    'lgbm_commodity': 'models/pppq/lgbm_target_commodity_score_regressor.txt'
}
```

---

## Data Flow Diagram (v2.0.0)

```
┌─────────────────────────────────────────────────────────────────────┐
│  External Data Sources                                              │
├─────────────────────────────────────────────────────────────────────┤
│  FRED    Yahoo Finance    World Bank    CoinGecko    BLS (NEW)     │
│  └─Econ   └─Prices        └─Global M2   └─Supply    └─Eggs/Milk    │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TASK 1: FETCH NEW DATA                                             │
│  └─ Filters to dates > last_date                                    │
│  └─ Features engineering (39 features with commodities)             │
│  └─ Appends to final_consolidated_dataset.csv                       │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  data/raw/final_consolidated_dataset.csv                            │
│  (Complete historical + new rows with commodity data)               │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TASK 2: PREPROCESS DATA (v2.0.0)                                   │
│  └─ Asset-level feature extraction (39 features)                    │
│  └─ Calculate 8 component score targets                             │
│  └─ PPP_Q classification labels                                     │
│  └─ Time-based splits (train/val/test)                              │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  data/processed/pppq/                                               │
│  ├─ train/pppq_train.csv    (2015-2022, 39 features, 9 targets)   │
│  ├─ val/pppq_val.csv        (2023-06, 39 features, 9 targets)     │
│  └─ test/pppq_test.csv      (2023-12, 39 features, 9 targets)     │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TASK 3: TRAIN MULTI-OUTPUT MODELS (v2.0.0)                         │
│  └─ 2 Classification Models (LightGBM + XGBoost ensemble)           │
│  └─ 8 Component Regression Models (LightGBM)                        │
│  └─ Total: 10 models trained                                        │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  models/pppq/ (10 model files)                                      │
│  ├─ lgbm_classifier.txt                                             │
│  ├─ xgb_classifier.json                                             │
│  ├─ lgbm_target_*_score_regressor.txt (8 regressors)               │
│  ├─ training_metrics_v2.json (NEW - v2.0.0 metrics)                │
│  └─ model_registry.json (MLflow tracking)                           │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TASK 4: EVALUATE & VERSION                                         │
│  └─ Classification F1: 96.30%                                       │
│  └─ Component Avg R²: 99.30%                                        │
│  └─ Compare with previous best                                      │
│  └─ Register with MLflow if better                                  │
│  └─ Decide auto-deployment                                          │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TASK 5: SEND NOTIFICATIONS (v2.0.0)                                │
│  └─ Email to ba8616127@gmail.com                                    │
│  └─ Metrics: 10 models, F1=96.3%, R²=99.3%, commodity score        │
│  └─ New data stats, deployment info                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Monitoring & Troubleshooting

### Pipeline Logs
```bash
# View recent pipeline logs
cat logs/pipeline.log

# Follow real-time logs (if running)
tail -f logs/pipeline.log
```

### Check Data Freshness
```bash
# Last date in consolidated dataset
python -c "
import pandas as pd
df = pd.read_csv('data/raw/final_consolidated_dataset.csv')
print(f'Total rows: {len(df)}')
print(f'Date range: {df[\"Date\"].min()} to {df[\"Date\"].max()}')
print(f'Features: {len(df.columns)}')
"
```

### Check Model Performance (v2.0.0)
```bash
# View v2.0.0 metrics
cat models/pppq/training_metrics_v2.json | python -m json.tool

# Validate all 10 models exist
python -c "
from pathlib import Path
models_dir = Path('models/pppq')
required = [
    'lgbm_classifier.txt',
    'xgb_classifier.json',
    'lgbm_target_real_pp_score_regressor.txt',
    'lgbm_target_volatility_score_regressor.txt',
    'lgbm_target_cycle_score_regressor.txt',
    'lgbm_target_growth_score_regressor.txt',
    'lgbm_target_consistency_score_regressor.txt',
    'lgbm_target_recovery_score_regressor.txt',
    'lgbm_target_risk_adjusted_score_regressor.txt',
    'lgbm_target_commodity_score_regressor.txt'
]
missing = [m for m in required if not (models_dir / m).exists()]
print(f'✓ All 10 models exist' if not missing else f'❌ Missing: {missing}')
"
```

### Common Issues (v2.0.0)

| Issue | Solution |
|-------|----------|
| **No new data detected** | Check FRED/Yahoo APIs are accessible. Date parsing issue? |
| **Missing commodity data** | Verify BLS API key or update manual commodity prices |
| **Preprocessing fails** | Check 39 features exist. Missing egg/milk columns? |
| **Model training slow** | Expected ~45 seconds for 10 models. Check hardware. |
| **Missing models after training** | Run `python src/models/pppq_multi_output_model.py` manually |
| **Email not sending** | Verify SMTP credentials in .env. Gmail needs app password. |
| **GitHub Actions fails** | Check secrets are set. FRED_API_KEY in repo settings? |
| **Git LFS errors** | Install git-lfs: `git lfs install`, track models: `git lfs track "models/pppq/*.txt"` |

---

## Performance Metrics (v2.0.0)

### Typical Pipeline Runtime
- **Data Ingestion:** 2-5 minutes (depends on API response times)
- **Preprocessing:** 1-2 minutes (70 assets × dates, 39 features)
- **Model Training:** 45-90 seconds (10 models with early stopping)
- **Evaluation:** 30 seconds
- **Notifications:** 5 seconds
- **Total:** ~5-10 minutes (much faster than v1.x!)

### Model Performance (v2.0.0 Production)

**Classification**:
- **Ensemble Macro F1:** 96.30% (96.5% LGBM + 96.7% XGB)
- **Accuracy:** 96.5%
- **Balanced Accuracy:** 96.2%

**Regression (Component Scores)**:
- **Average R²:** 99.3% across 8 regressors
- **Best Performer:** Real PP Score (99.5% R²)
- **NEW:** Commodity Score (99.4% R²)
- **Average RMSE:** 1.2 points (on 0-100 scale)

### Improvement vs v1.x

| Metric | v1.x | v2.0.0 | Improvement |
|--------|------|--------|-------------|
| Classification F1 | 78% | 96.3% | +18.3% |
| Num Models | 3 | 10 | +233% |
| Features | 18 | 39 | +117% |
| Interpretability | Low | High | Component scores! |
| Horizon Awareness | No | Yes | Dynamic |
| Commodity Tracking | No | Yes | Eggs & milk |

---

## Files Modified by Pipeline (v2.0.0)

Each run updates:
```
data/raw/
  └─ final_consolidated_dataset.csv          ← NEW ROWS APPENDED

data/processed/pppq/
  ├─ train/pppq_train.csv                    ← Updated (39 features)
  ├─ val/pppq_val.csv                        ← Updated (39 features)
  ├─ test/pppq_test.csv                      ← Updated (39 features)
  ├─ pppq_features.json                      ← Feature list (39)
  ├─ pppq_summary.json                       ← Dataset stats
  └─ pppq_thresholds.json                    ← Dynamic thresholds

models/pppq/
  ├─ lgbm_classifier.txt                     ← NEW classification model
  ├─ xgb_classifier.json                     ← NEW classification model
  ├─ lgbm_target_real_pp_score_regressor.txt           ← NEW regressor
  ├─ lgbm_target_volatility_score_regressor.txt       ← NEW regressor
  ├─ lgbm_target_cycle_score_regressor.txt            ← NEW regressor
  ├─ lgbm_target_growth_score_regressor.txt           ← NEW regressor
  ├─ lgbm_target_consistency_score_regressor.txt      ← NEW regressor
  ├─ lgbm_target_recovery_score_regressor.txt         ← NEW regressor
  ├─ lgbm_target_risk_adjusted_score_regressor.txt    ← NEW regressor
  ├─ lgbm_target_commodity_score_regressor.txt        ← NEW regressor
  ├─ label_encoder.pkl                       ← Class encoder
  ├─ feature_columns.json                    ← 39 features
  ├─ model_registry.json                     ← MLflow history (v2.0.0)
  ├─ training_metrics_v2.json                ← v2.0.0 metrics
  └─ training_summary.json                   ← Training summary

logs/
  └─ pipeline.log                            ← Execution log
```

---

## Next Steps & Improvements

### Completed (v2.0.0) ✅
- [x] Multi-output architecture (2 classifiers + 8 regressors)
- [x] Horizon-aware predictions (1Y-10Y dynamic adjustments)
- [x] Commodity features (eggs & milk purchasing power)
- [x] Ensemble classification (LightGBM + XGBoost)
- [x] Component score explainability
- [x] Dynamic class thresholds (asset category + horizon)
- [x] Model versioning with MLflow
- [x] Automated workflows (GitHub Actions)
- [x] Git LFS for model files

### Future Enhancements
- [ ] Real-time BLS API integration (automate egg/milk price fetching)
- [ ] SHAP values for feature importance explanations
- [ ] Hyperparameter tuning with Optuna (auto-optimize model params)
- [ ] Data drift detection (alert if input distribution changes)
- [ ] A/B testing framework (compare v2 vs v3 in production)
- [ ] Kubernetes deployment (scale horizontally)
- [ ] Additional commodity baskets (bread, housing, gasoline)
- [ ] Sentiment analysis (news + social media for market cycle)
- [ ] On-chain metrics (Bitcoin network health indicators)

---

## Documentation References

For deeper understanding of v2.0.0:

- **Complete ML System Guide**: [docs/COMPLETE_ML_SYSTEM_GUIDE.md](docs/COMPLETE_ML_SYSTEM_GUIDE.md)
  - Every model explained step-by-step
  - Feature engineering rationale
  - Real-world examples
  - Dynamic threshold logic

- **Workflow Updates**: [docs/WORKFLOW_UPDATES_v2.md](docs/WORKFLOW_UPDATES_v2.md)
  - GitHub Actions changes
  - Prefect orchestration updates
  - Deployment checklist

- **Prefect v2.0.0**: [docs/PREFECT_V2_UPDATE.md](docs/PREFECT_V2_UPDATE.md)
  - Multi-output training integration
  - Model validation
  - Notification changes

---

**Last Updated:** December 17, 2024 (v2.0.0)
**Maintainer:** Bilal Ahmad Sheikh (GIKI)
**Pipeline Type:** Incremental | **Frequency:** Every 15 days
**Notification:** Email to ba8616127@gmail.com
**Architecture:** Multi-Output (2 Classifiers + 8 Regressors)
**Performance:** 96.3% Classification F1, 99.3% Component R²
