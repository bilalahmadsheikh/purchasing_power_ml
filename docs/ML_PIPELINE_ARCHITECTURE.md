# PPP-Q ML Pipeline Architecture - v2.0.0

**Complete End-to-End Machine Learning Pipeline Documentation**

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Data Flow](#data-flow)
4. [Components](#components)
5. [Workflows](#workflows)
6. [Model Training Pipeline](#model-training-pipeline)
7. [Prediction Pipeline](#prediction-pipeline)
8. [Automation & Orchestration](#automation--orchestration)
9. [Monitoring & Validation](#monitoring--validation)
10. [Deployment](#deployment)

---

## Overview

The PPP-Q ML Pipeline is a **multi-output machine learning system** that predicts both:
1. **Asset Classification** (A/B/C/D tiers) - 96.30% Macro-F1
2. **Component Scores** (8 regression targets) - 99.3% average R²

### Key Features
- ✅ **Horizon-Aware Predictions** (1Y to 10Y investment horizons)
- ✅ **Real Commodity Tracking** (Eggs/Milk purchasing power)
- ✅ **Zero Hardcoded Logic** (Pure ML-predicted component scores)
- ✅ **Automated Retraining** (Daily/weekly data updates)
- ✅ **CI/CD Integration** (GitHub Actions workflows)
- ✅ **Model Versioning** (MLflow + Git LFS)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PPP-Q ML PIPELINE v2.0.0                        │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Data        │      │  Feature     │      │  Model       │
│  Collection  │─────▶│  Engineering │─────▶│  Training    │
│              │      │              │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
      │                      │                      │
      │                      │                      │
      ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  External    │      │  39 Features │      │  10 Models   │
│  APIs:       │      │  ───────────  │      │  ──────────  │
│  • CoinGecko │      │  • PP Mults  │      │  • 2 Clf.    │
│  • Yahoo Fin │      │  • Vol/Risk  │      │  • 8 Reg.    │
│  • FRED      │      │  • Egg/Milk  │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
                                                    │
                                                    │
                                                    ▼
                                            ┌──────────────┐
                                            │  Validation  │
                                            │  & Testing   │
                                            └──────────────┘
                                                    │
                                                    │
                                                    ▼
                                            ┌──────────────┐
                                            │  Deployment  │
                                            │  • API       │
                                            │  • Streamlit │
                                            └──────────────┘
```

---

## Data Flow

### 1. Data Collection (`src/data/data_collection.py`)

```python
# External Data Sources
├── CoinGecko API         # Crypto prices, market caps
├── Yahoo Finance         # Stocks, ETFs, indices
├── FRED API              # CPI, inflation, commodities
└── Manual Data           # Egg/milk prices (BLS)
```

**Collected Metrics**:
- Price data (OHLCV)
- Market capitalization
- Trading volume
- CPI (Consumer Price Index)
- Egg prices ($/dozen)
- Milk prices ($/gallon)

**Output**: `data/raw/final_consolidated_dataset.csv`

---

### 2. Feature Engineering (`src/data/preprocessing_pppq.py`)

Transforms raw data into 39 ML-ready features:

#### **Base Features (18 original)**
| Feature | Description | Type |
|---------|-------------|------|
| `PP_Multiplier_1Y` | 1-year purchasing power multiplier | Float |
| `PP_Multiplier_5Y` | 5-year purchasing power multiplier | Float |
| `PP_Multiplier_10Y` | 10-year purchasing power multiplier | Float |
| `Real_Return_1Y` | Real return (%) after CPI adjustment | Float |
| `Real_Return_5Y` | 5-year real return | Float |
| `Real_Return_10Y` | 10-year real return | Float |
| `Volatility_90D` | 90-day rolling volatility (%) | Float |
| `Sharpe_Ratio_1Y` | Risk-adjusted return (1Y) | Float |
| `Sharpe_Ratio_5Y` | Risk-adjusted return (5Y) | Float |
| `Max_Drawdown` | Maximum peak-to-trough decline (%) | Float |
| `Distance_From_ATH_Pct` | Distance from all-time high (%) | Float |
| `Distance_From_MA_200D_Pct` | Distance from 200-day MA (%) | Float |
| `Days_Since_ATH` | Days since ATH | Integer |
| `Market_Cap_Saturation_Pct` | Market maturity indicator | Float |
| `Calmar_Ratio` | Return/Max Drawdown | Float |
| `Sortino_Ratio` | Downside risk-adjusted return | Float |
| `Recovery_Strength` | Speed of recovery from drawdowns | Float |
| `Return_Consistency` | Consistency across horizons | Float |

#### **New Commodity Features (5 new in v2.0.0)**
| Feature | Description | Type |
|---------|-------------|------|
| `Eggs_Per_100USD` | Dozens of eggs $100 can buy | Float |
| `Milk_Gallons_Per_100USD` | Gallons of milk $100 can buy | Float |
| `Real_Return_Eggs_1Y` | Return measured in egg purchasing power | Float |
| `Real_Return_Milk_1Y` | Return measured in milk purchasing power | Float |
| `Real_Commodity_Basket_Return_1Y` | Avg(eggs, milk) return | Float |

#### **Target Variables (8 component scores)**
| Target | Description | R² Score |
|--------|-------------|----------|
| `Target_Real_PP_Score` | Real purchasing power score (0-100) | 0.998 |
| `Target_Volatility_Score` | Volatility risk score (0-100) | 0.977 |
| `Target_Cycle_Score` | Market cycle score (0-100) | 0.988 |
| `Target_Growth_Score` | Growth potential score (0-100) | 1.000 |
| `Target_Consistency_Score` | Consistency score (0-100) | 0.986 |
| `Target_Recovery_Score` | Recovery speed score (0-100) | 0.997 |
| `Target_Risk_Adjusted_Score` | Risk-adjusted score (0-100) | 0.999 |
| `Target_Commodity_Score` | Commodity PP score (0-100) | 1.000 |

**Output**:
- `data/processed/pppq/train/pppq_train.csv` (2010-2021)
- `data/processed/pppq/val/pppq_val.csv` (2022-2023)
- `data/processed/pppq/test/pppq_test.csv` (2024-2025)

---

### 3. Model Training (`src/models/pppq_multi_output_model.py`)

#### **Multi-Output Training Strategy**

```python
# Train 10 models in total:
# 1. Classification models (2)
# 2. Component score regressors (8)

Training Pipeline:
├── Load preprocessed data (39 features + 8 targets)
├── Split train/val/test (time-based)
│
├── Classification Training
│   ├── LightGBM Classifier
│   │   └── Output: A_PRESERVER, B_PARTIAL, C_ERODER, D_DESTROYER
│   └── XGBoost Classifier
│       └── Output: A_PRESERVER, B_PARTIAL, C_ERODER, D_DESTROYER
│
└── Component Score Training (8 parallel LightGBM regressors)
    ├── Real PP Score Regressor       (R² = 0.998)
    ├── Volatility Score Regressor    (R² = 0.977)
    ├── Cycle Score Regressor         (R² = 0.988)
    ├── Growth Score Regressor        (R² = 1.000) ✨
    ├── Consistency Score Regressor   (R² = 0.986)
    ├── Recovery Score Regressor      (R² = 0.997)
    ├── Risk-Adjusted Score Regressor (R² = 0.999)
    └── Commodity Score Regressor     (R² = 1.000) ✨
```

#### **Training Configuration**

**LightGBM Classifier**:
```python
{
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 8,
    'min_data_in_leaf': 50,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1
}
```

**XGBoost Classifier**:
```python
{
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'num_class': 4,
    'eval_metric': 'mlogloss',
    'early_stopping_rounds': 20
}
```

**LightGBM Regressors** (8 models):
```python
{
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 6,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.05,
    'lambda_l2': 0.05,
    'verbose': -1
}
```

**Training Time**: ~45 seconds (all 10 models)

---

## Prediction Pipeline

### Step-by-Step Prediction Flow

```
1. User Request
   ↓
2. Load Asset Data (latest window)
   ↓
3. Horizon-Aware Feature Preparation ⭐
   │
   ├── Short-term (<2Y)
   │   • Stricter volatility penalties
   │   • Higher cycle position sensitivity
   │   • Conservative growth adjustments
   │
   ├── Medium-term (2-5Y)
   │   • Balanced approach
   │   • Standard multipliers
   │
   └── Long-term (>5Y)
       • Volatility tolerance (time diversification)
       • Growth potential boost
       • Drawdown recovery consideration
   ↓
4. ML Prediction
   │
   ├── Classification (Ensemble)
   │   • LightGBM: 95.94% F1
   │   • XGBoost: 96.50% F1
   │   • Ensemble: 96.30% F1 ⭐
   │   └── Output: A_PRESERVER / B_PARTIAL / C_ERODER / D_DESTROYER
   │
   └── Component Scores (8 Regressors)
       • Real PP Score: 0-100
       • Volatility Score: 0-100
       • Cycle Score: 0-100
       • Growth Score: 0-100
       • Consistency Score: 0-100
       • Recovery Score: 0-100
       • Risk-Adjusted Score: 0-100
       • Commodity Score: 0-100
       └── Composite: Weighted average (0-100)
   ↓
5. Generate Insights
   │
   ├── Strengths (Top 3)
   ├── Weaknesses (Top 3)
   ├── Current Status
   │   • Volatility assessment
   │   • Cycle position
   │   • Entry signal
   │   • Growth potential
   └── Metrics
       • PP multipliers
       • Sharpe ratios
       • Drawdowns
       • Real returns
   ↓
6. Return PredictionOutput (JSON)
```

### Horizon-Aware Feature Adjustments

**Example: Bitcoin 1Y vs 10Y**

| Feature | Base Value | 1Y Horizon | 10Y Horizon | Adjustment Logic |
|---------|-----------|------------|-------------|------------------|
| `PP_Multiplier_5Y` | 2.50x | 0.50x | 5.00x | `base * (horizon / 5.0)` |
| `Volatility_90D` | 60% | 60% | 36% | `base * vol_decay` (time diversification) |
| `Distance_From_ATH` | -30% | -36% | -24% | Stricter for short-term |
| `Sharpe_Ratio_5Y` | 1.2 | 1.2 | 2.16 | `base * (1 + (h-1)*0.12)` |
| `Max_Drawdown` | 75% | 75% | 60% | More tolerance for long-term |

---

## Workflows

### GitHub Actions Automation

```
.github/workflows/
├── ci-cd.yml                  # Main CI/CD pipeline
├── ml-validation.yml          # Model performance tests
├── data-validation.yml        # Data quality checks
├── model-training.yml         # Automated retraining
├── automated-pipeline.yml     # End-to-end pipeline
├── integration-tests.yml      # API integration tests
└── release.yml                # Release automation
```

---

### 1. Data Collection Workflow

**File**: `.github/workflows/automated-pipeline.yml`

```yaml
name: Automated ML Pipeline

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly (Sunday midnight)
  workflow_dispatch:

jobs:
  data-collection:
    runs-on: ubuntu-latest
    steps:
      - name: Fetch latest data
        run: python src/data/data_collection.py
        env:
          COINGECKO_API_KEY: ${{ secrets.COINGECKO_API_KEY }}
          FRED_API_KEY: ${{ secrets.FRED_API_KEY }}

      - name: Upload raw data
        uses: actions/upload-artifact@v3
        with:
          name: raw-data
          path: data/raw/
```

**Triggers**:
- ⏰ Scheduled (Weekly on Sundays)
- 🔘 Manual dispatch

**Steps**:
1. Fetch crypto data (CoinGecko)
2. Fetch stock data (Yahoo Finance)
3. Fetch CPI data (FRED)
4. Consolidate into `final_consolidated_dataset.csv`

---

### 2. Preprocessing Workflow

**Triggered After**: Data collection

```yaml
  preprocessing:
    needs: data-collection
    runs-on: ubuntu-latest
    steps:
      - name: Run preprocessing
        run: python src/data/preprocessing_pppq.py

      - name: Validate features
        run: python src/ml_testing/data_validation.py

      - name: Check for drift
        run: python src/ml_testing/drift_detection.py
```

**Steps**:
1. Calculate PP multipliers
2. Calculate risk metrics
3. **NEW**: Calculate egg/milk features
4. Generate 8 component score targets
5. Split train/val/test
6. Validate data quality
7. Detect distribution drift

---

### 3. Model Training Workflow

**File**: `.github/workflows/model-training.yml`

```yaml
name: Model Training

on:
  workflow_dispatch:
  schedule:
    - cron: '0 2 * * 1'  # Weekly (Monday 2 AM)

jobs:
  train-models:
    runs-on: ubuntu-latest
    steps:
      - name: Train multi-output models
        run: python src/models/pppq_multi_output_model.py

      - name: Validate model performance
        run: python src/ml_testing/model_validation.py

      - name: Save models
        run: |
          git lfs track "models/pppq/*.txt"
          git lfs track "models/pppq/*.json"
          git add models/pppq/
          git commit -m "feat: retrained models $(date +%Y-%m-%d)"
          git push
```

**Steps**:
1. Train 2 classifiers (LightGBM + XGBoost)
2. Train 8 component regressors (LightGBM)
3. Validate Macro-F1 ≥ 90%
4. Validate Component R² ≥ 0.95
5. Save models to Git LFS
6. Push to repository

---

### 4. CI/CD Workflow

**File**: `.github/workflows/ci-cd.yml`

```yaml
name: CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: docker-compose up -d
```

**Triggers**:
- Push to `main` or `develop`
- Pull requests to `main`

**Steps**:
1. Lint code (flake8, black)
2. Run unit tests
3. Run integration tests
4. Check code coverage (>80%)
5. Deploy if on `main` branch

---

## Monitoring & Validation

### Data Validation (`src/ml_testing/data_validation.py`)

**Checks**:
- ✅ No missing values in critical columns
- ✅ No duplicate rows
- ✅ Feature ranges within expected bounds
- ✅ Target distribution not skewed
- ✅ Sufficient samples per class

**Thresholds**:
```python
MAX_MISSING_RATIO = 0.05  # Max 5% missing
MIN_SAMPLES_PER_CLASS = 100
FEATURE_BOUNDS = {
    'PP_Multiplier_5Y': (0.5, 10.0),
    'Volatility_90D': (0, 150),
    'Sharpe_Ratio_5Y': (-2, 5)
}
```

---

### Drift Detection (`src/ml_testing/drift_detection.py`)

**Methods**:
1. **Kolmogorov-Smirnov Test** (numerical features)
2. **Chi-Square Test** (categorical features)
3. **Population Stability Index (PSI)**

**Alert Thresholds**:
```python
DRIFT_THRESHOLD = 0.05  # p-value < 0.05 = drift detected
PSI_THRESHOLD = 0.2     # PSI > 0.2 = significant drift
```

**Monitored Features**:
- PP multipliers
- Volatility metrics
- Market cap saturation
- **NEW**: Egg/milk purchasing power

---

### Model Validation (`src/ml_testing/model_validation.py`)

**Performance Thresholds**:
```python
MIN_MACRO_F1 = 0.90      # 90% minimum F1
MIN_ACCURACY = 0.85      # 85% minimum accuracy
MIN_COMPONENT_R2 = 0.95  # 95% minimum R² for regressors
```

**Validation Tests**:
1. Classification F1 score ≥ 90%
2. Component R² scores ≥ 95%
3. No class with F1 < 80%
4. Prediction time < 100ms

---

## Deployment

### FastAPI Backend (`src/api/`)

```python
# Main endpoints
GET  /                    # Health check
POST /predict             # Single prediction
POST /compare             # Compare multiple assets
GET  /assets              # List available assets
GET  /model/info          # Model metadata
GET  /historical/{asset}  # Historical predictions
GET  /data/quality        # Data quality metrics
```

**Model Loading** (Singleton Pattern):
```python
class ModelManager:
    _instance = None

    def __init__(self):
        self.lgbm_classifier = None
        self.xgb_classifier = None
        self.component_models = {}  # 8 regressors
        self.encoder = None
        self.features = []
```

---

### Streamlit App (`streamlit_app/app.py`)

**Features**:
- 🎯 Single asset analysis
- 📊 Multi-asset comparison
- 📈 Correlation analysis
- 🔧 Data pipeline (retrain models)
- 📚 Documentation

**Model Loading**:
- Primary: GitHub raw URLs
- Fallback: Local files
- Data: Google Drive + Local

---

## File Structure

```
purchasing_power_ml/
│
├── data/
│   ├── raw/
│   │   └── final_consolidated_dataset.csv
│   └── processed/pppq/
│       ├── train/pppq_train.csv (2010-2021)
│       ├── val/pppq_val.csv (2022-2023)
│       └── test/pppq_test.csv (2024-2025)
│
├── models/pppq/
│   ├── lgbm_classifier.txt (2.1 MB)
│   ├── xgb_classifier.json (2.9 MB)
│   ├── lgbm_target_real_pp_score_regressor.txt
│   ├── lgbm_target_volatility_score_regressor.txt
│   ├── lgbm_target_cycle_score_regressor.txt
│   ├── lgbm_target_growth_score_regressor.txt
│   ├── lgbm_target_consistency_score_regressor.txt
│   ├── lgbm_target_recovery_score_regressor.txt
│   ├── lgbm_target_risk_adjusted_score_regressor.txt
│   ├── lgbm_target_commodity_score_regressor.txt
│   ├── feature_columns.json (39 features)
│   └── component_targets.json (8 targets)
│
├── src/
│   ├── api/
│   │   ├── main.py (FastAPI app)
│   │   ├── predict_ml.py (ML prediction logic)
│   │   ├── schemas.py (Pydantic models)
│   │   └── config.py (Settings)
│   │
│   ├── data/
│   │   ├── data_collection.py
│   │   └── preprocessing_pppq.py
│   │
│   ├── models/
│   │   └── pppq_multi_output_model.py
│   │
│   ├── ml_testing/
│   │   ├── data_validation.py
│   │   ├── drift_detection.py
│   │   └── model_validation.py
│   │
│   └── pipelines/
│       ├── prefect_flows.py
│       ├── model_registry.py
│       └── pipeline_config.py
│
├── tests/
│   ├── test_api.py
│   ├── test_ml_validation.py
│   └── test_new_endpoints.py
│
├── .github/workflows/
│   ├── ci-cd.yml
│   ├── ml-validation.yml
│   ├── model-training.yml
│   └── automated-pipeline.yml
│
└── docs/
    ├── ML_PIPELINE_ARCHITECTURE.md (this file)
    ├── API_DOCUMENTATION.md
    └── MODEL_CHANGELOG_v2.md
```

---

## Version History

### v2.0.0 (Current)
- ✅ ML-powered component scores (99.3% R²)
- ✅ Egg/milk commodity features
- ✅ Horizon-aware predictions
- ✅ 96.30% classification accuracy

### v1.2.0 (Previous)
- Hardcoded component scoring
- 90.35% classification accuracy
- 18 features

---

## Next Steps / Roadmap

### v2.1.0 (Planned)
- [ ] SHAP explanations for component scores
- [ ] Feature importance dashboards
- [ ] A/B testing framework
- [ ] Real-time data streaming (WebSocket)

### v3.0.0 (Future)
- [ ] Deep learning models (LSTM for time series)
- [ ] Reinforcement learning for portfolio optimization
- [ ] Multi-asset correlation predictions
- [ ] Sentiment analysis from news/social

---

## Contact & Support

**Author**: Bilal Ahmad Sheikh (GIKI)
**Version**: v2.0.0
**Last Updated**: 2024-12-17

**GitHub**: https://github.com/bilalahmadsheikh/purchasing_power_ml
**Documentation**: `/docs`
**Issues**: https://github.com/bilalahmadsheikh/purchasing_power_ml/issues

---

**End of ML Pipeline Architecture Documentation**
