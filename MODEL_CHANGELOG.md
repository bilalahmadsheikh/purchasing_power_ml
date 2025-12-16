# PPP-Q Model Changelog

This file tracks all model versions, performance changes, and significant updates.

---

## Model Version History

### v1.2.0 - 2025-12-16 (Ensemble Model Support)
**Best Model:** Ensemble (LightGBM + XGBoost)  
**Macro F1:** ~90.35% (averaged)  
**Features:** 17 PPP-Q features

#### New Features:
- **Ensemble Model Support** - Average of LightGBM + XGBoost probabilities
- **Model Type Selection** - API now supports `lgbm`, `xgb`, or `ensemble` (default)
- **Dynamic Weights** - Component weights adjust based on investment horizon
- **Threshold-Based Classification** - Final class based on composite score thresholds

#### Model Types Available:
| Type | Description | Speed | Robustness |
|------|-------------|-------|------------|
| `lgbm` | LightGBM only | Fastest | Good |
| `xgb` | XGBoost only | Fast | Good |
| `ensemble` | Average of both | Moderate | Best (default) |

#### Dynamic Weight System:
| Horizon | PP Score | Volatility | Cycle | Growth | Consistency | Recovery | Risk-Adj |
|---------|----------|------------|-------|--------|-------------|----------|----------|
| <2Y | 25% | 25% | 20% | 10% | 10% | 10% | 0% |
| 2-5Y | 25% | 20% | 15% | 15% | 10% | 10% | 5% |
| 5Y+ | 20% | 15% | 10% | 20% | 10% | 15% | 10% |

---

### v1.1.0 - 2025-12-16 (Balanced Classification)
**Best Model:** LightGBM  
**Macro F1:** 90.28%  
**Training Data:** 2010-01-01 to 2021-12-31  
**Validation Data:** 2022-01-01 to 2023-12-31  
**Test Data:** 2024-01-01 to 2025-12-31

#### Classification Thresholds (Updated):
| Class | Score Threshold | Description |
|-------|-----------------|-------------|
| A_PRESERVER | ≥ 65 | Strong PP preservation + growth |
| B_PARTIAL | 55-64 | Adequate PP preservation |
| C_ERODER | 42-54 | Marginal, may lose to inflation |
| D_DESTROYER | < 42 | Significant PP destruction |

#### Model Performance by Class:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| A_PRESERVER | 0.98 | 0.92 | 0.95 | 3,065 |
| B_PARTIAL | 0.91 | 0.87 | 0.89 | 4,141 |
| C_ERODER | 0.83 | 0.95 | 0.89 | 2,899 |
| D_DESTROYER | 0.95 | 0.83 | 0.88 | 620 |

#### Class Distribution:
| Split | A_PRESERVER | B_PARTIAL | C_ERODER | D_DESTROYER |
|-------|-------------|-----------|----------|-------------|
| Train | 25.1% | 25.7% | 35.7% | 13.5% |
| Val | 29.0% | 33.3% | 35.5% | 2.1% |
| Test | 28.6% | 38.6% | 27.0% | 5.8% |

---

### v1.0.0 - 2024-12-16 (Initial Release)
**Best Model:** XGBoost  
**Macro F1:** 97.10%  
**Note:** Initial version had unbalanced classification thresholds

---

## Update Log

### 2025-12-16 - Ensemble Model Support
**Feature:** Added ensemble model combining LightGBM + XGBoost  
**Implementation:** Average probabilities from both models  
**API Change:** New `model_type` parameter: `lgbm`, `xgb`, `ensemble` (default)  
**Files Changed:** `src/api/predict.py`, `src/api/main.py`, `src/api/schemas.py`  
**Impact:** More robust predictions with ensemble averaging

### 2025-12-16 - Dynamic Weight System
**Feature:** Component weights now adjust based on investment horizon  
**Short-term (<2Y):** Emphasizes stability (25% volatility, 20% cycle)  
**Long-term (5Y+):** Emphasizes growth (20% growth, 15% recovery)  
**Files Changed:** `src/api/predict.py`  
**Impact:** More accurate predictions for different investment timeframes

### 2025-12-16 - Threshold-Based Classification
**Feature:** Classification now uses composite score thresholds directly  
**Thresholds:** A≥65, B≥55, C≥42, D<42  
**Files Changed:** `src/api/predict.py`  
**Impact:** Classifications exactly match composite scores

### 2025-12-16 - Feature Engineering Fix
**Issue:** Incremental data updates were causing NaN/0 values for rolling features  
**Root Cause:** Feature engineering was applied only to new rows without historical context  
**Solution:** Combined raw data first, then applied feature engineering to full dataset  
**Files Changed:** `src/pipelines/prefect_flows.py`  
**Impact:** All rolling calculations (returns, volatility) now have proper historical context

### 2024-12-16 - Prediction API Fix
**Issue:** API returned "features in data (1) != training data (17)"  
**Root Cause:** feature_columns.json saved as `{'features': [...]}` but API expected list  
**Solution:** Updated predict.py to handle both dict and list formats  
**Files Changed:** `src/api/predict.py`

---

## Deployment History

| Date | Version | Model | Macro F1 | Deployed | Notes |
|------|---------|-------|----------|----------|-------|
| 2025-12-16 | v1.2.0 | Ensemble | ~90.35% | Yes | Ensemble + dynamic weights |
| 2025-12-16 | v1.1.0 | LightGBM | 90.28% | Yes | Balanced classification |
| 2025-12-16 | v1.0.0 | XGBoost | 97.10% | Yes | Initial production deployment |

---

## Performance Benchmarks

### Assets by Classification (Latest)
| Asset | Class | Confidence | Real Return 5Y | PP Multiplier 5Y |
|-------|-------|------------|----------------|------------------|
| Bitcoin | A_PRESERVER | 95.8% | 326.56% | 4.27 |
| Ethereum | A_PRESERVER | 93.2% | 285.4% | 3.85 |
| Gold | B_PARTIAL | 87.5% | 12.3% | 1.12 |
| S&P 500 | B_PARTIAL | 91.2% | 45.6% | 1.46 |
| Cash/USD | C_ERODER | 94.1% | -15.2% | 0.85 |

---

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PPP-Q Classification v1.2                 │
├─────────────────────────────────────────────────────────────┤
│  Data Pipeline                                               │
│  ├── Data Ingestion (FRED, Yahoo Finance)                   │
│  ├── Feature Engineering (17 PPP-Q features)                │
│  └── Train/Val/Test Split (temporal, no leakage)            │
├─────────────────────────────────────────────────────────────┤
│  Model Ensemble                                              │
│  ├── LightGBM (90.28% F1)                                   │
│  ├── XGBoost (89.44% F1)                                    │
│  └── Ensemble = (LightGBM + XGBoost) / 2                    │
├─────────────────────────────────────────────────────────────┤
│  Classification (Composite Score Thresholds)                 │
│  ├── A_PRESERVER (Score ≥ 65)                               │
│  ├── B_PARTIAL (Score 55-64)                                │
│  ├── C_ERODER (Score 42-54)                                 │
│  └── D_DESTROYER (Score < 42)                               │
├─────────────────────────────────────────────────────────────┤
│  API Endpoints                                               │
│  ├── POST /predict (single asset)                           │
│  ├── POST /predict/batch (multiple assets)                  │
│  └── model_type: lgbm | xgb | ensemble                      │
└─────────────────────────────────────────────────────────────┘
```

---

## How to Update This Log

After each pipeline run that deploys a new model:

1. Check `reports/pppq/training_summary.json` for metrics
2. Add new entry under "Update Log" with date and changes
3. Update "Deployment History" table if model was deployed
4. Update "Performance Benchmarks" if significant changes

The pipeline automatically logs to `logs/pipeline.log` for detailed run history.
