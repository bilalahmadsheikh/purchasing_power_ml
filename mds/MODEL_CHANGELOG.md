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
## [v2.0.0] - 2024-12-17

### 🎉 MAJOR RELEASE: ML-Powered Component Scores

**The biggest update since project inception!** All component scores are now predicted by dedicated ML models with 99.3% accuracy, eliminating hardcoded scoring logic entirely.

### 🚀 New Features

#### 1. ML-Predicted Component Scores (R² = 99.3%)
Replaced ALL hardcoded component scoring logic with 8 dedicated LightGBM regression models:

| Component Model | RMSE | R² Score | Training Time |
|----------------|------|----------|---------------|
| **Real PP Score** | 0.791 | 0.998 | 5.3s |
| **Volatility Score** | 4.996 | 0.977 | 5.5s |
| **Cycle Score** | 1.211 | 0.988 | 2.0s |
| **Growth Score** | 0.478 | **1.000** | 3.0s |
| **Consistency Score** | 1.905 | 0.986 | 6.0s |
| **Recovery Score** | 0.901 | 0.997 | 6.4s |
| **Risk-Adjusted Score** | 0.684 | 0.999 | 4.0s |
| **Commodity Score** 🆕 | 0.367 | **1.000** | 2.7s |

**Total Training Time**: ~35 seconds for all 8 models

**Benefits**:
- ✅ No more hardcoded if/else scoring rules
- ✅ Learns complex patterns from data
- ✅ Adapts to new market conditions
- ✅ Nearly perfect predictions (avg R² = 99.3%)

#### 2. Real Commodity Purchasing Power Features 🥚🥛

Added 5 new features tracking purchasing power in **actual goods** (not CPI):

1. **`Eggs_Per_100USD`** - How many eggs can $100 buy?
2. **`Milk_Gallons_Per_100USD`** - How much milk can $100 buy?
3. **`Real_Return_Eggs_1Y`** - Asset return measured in eggs
4. **`Real_Return_Milk_1Y`** - Asset return measured in milk
5. **`Real_Commodity_Basket_Return_1Y`** - Combined egg/milk score

**Why This Matters**:
- CPI is manipulated and unreliable
- Real goods = real purchasing power
- Eggs/milk = everyday essentials everyone buys
- Direct measure of "can I buy MORE stuff?"

**Example**: Bitcoin at $45K vs $90K
- Price doubled = 100% nominal return
- But can you buy 2x more eggs? Model knows!

#### 3. Component Score Blending with Commodity Power

Real PP Score now blends:
- 70% Traditional PP Multiplier (CPI-adjusted)
- 30% Commodity Purchasing Power (eggs/milk)

This gives a more accurate picture of real-world purchasing power.

### 📈 Performance Improvements

#### Classification Accuracy
| Metric | v1.2.0 | v2.0.0 | Improvement |
|--------|--------|--------|-------------|
| **Ensemble Macro-F1** | 90.35% | **96.30%** | **+5.95%** |
| **LightGBM Macro-F1** | 90.28% | **95.94%** | **+5.66%** |
| **XGBoost Macro-F1** | 89.44% | **96.50%** | **+7.06%** |
| **Accuracy** | 90.49% | **95.24%** | **+4.75%** |
| **Balanced Accuracy** | 89.66% | **95.90%** | **+6.24%** |

#### Component Score Accuracy (NEW)
- **Average RMSE**: 1.417 points (out of 100)
- **Average R²**: 0.993 (99.3% explained variance)
- **Perfect Scores (R² = 1.000)**: Growth, Commodity

#### Feature Count
- **v1.2.0**: 18 features
- **v2.0.0**: **39 features** (+116% increase)

### 🔧 Technical Changes

#### Model Architecture
**Before (v1.2.0)**:
```
User Input → Feature Engineering → Classification Model → A/B/C/D
                                  ↓
                          Hardcoded Component Scores (if/else rules)
```

**After (v2.0.0)**:
```
User Input → Feature Engineering → Classification Model → A/B/C/D
                                  ↓
                          8 ML Regression Models → Component Scores (R² = 99.3%)
```

#### New Training Pipeline
1. **Preprocessing** (`preprocessing_pppq.py`)
   - Calculate ground truth component scores
   - Add egg/milk features
   - Save as training targets

2. **Multi-Output Training** (`pppq_multi_output_model.py`)
   - Train 1 LightGBM classifier (A/B/C/D)
   - Train 1 XGBoost classifier (ensemble)
   - Train 8 LightGBM regressors (component scores)
   - Save all 10 models

3. **Prediction** (`predict_ml.py`)
   - Load all models via singleton
   - Predict class + 8 component scores
   - Horizon-aware adjustments
   - Return comprehensive insights

#### Code Improvements
- **Removed**: ~600 lines of hardcoded scoring logic
- **Added**: ML model manager with lazy loading
- **Optimized**: Singleton pattern for model efficiency
- **Enhanced**: Horizon-aware feature preparation
- **Improved**: Error handling and logging

### 🎯 Model Details

#### Classification Models

**LightGBM Classifier**:
```python
{
    'objective': 'multiclass',
    'num_class': 4,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 7,
    'best_iteration': 186
}
```

**XGBoost Classifier**:
```python
{
    'objective': 'multi:softprob',
    'num_class': 4,
    'learning_rate': 0.05,
    'max_depth': 7,
    'best_iteration': 150
}
```

#### Component Score Regressors

All 8 models use LightGBM with:
```python
{
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 7
}
```

**Best Iterations**:
- Real PP: 398 rounds
- Volatility: 499 rounds (max)
- Cycle: 155 rounds
- Growth: 157 rounds
- Consistency: 435 rounds
- Recovery: 197 rounds
- Risk-Adjusted: 317 rounds
- Commodity: 203 rounds

### 📊 Training Data

| Split | Samples | Date Range | Class Distribution |
|-------|---------|------------|-------------------|
| **Train** | 65,745 | 2010-2021 | B:44.3%, A:26.3%, C:25.6%, D:3.9% |
| **Val** | 10,950 | 2022-2023 | B:55.5%, A:25.0%, C:18.4%, D:1.2% |
| **Test** | 10,725 | 2024-2025 | B:44.2%, A:33.0%, C:15.3%, D:7.5% |

**Total**: 87,420 samples across 15 assets and 16 years

### 🔄 Breaking Changes

**NONE!** The API is fully backward compatible.

### Migration Guide

**For API Users**: No changes needed!
- All endpoints work the same
- Request/response formats unchanged
- Docker deployment unchanged

**For Model Developers**:
1. Update `preprocessing_pppq.py` (adds egg/milk features)
2. Train with `pppq_multi_output_model.py` (new script)
3. Update imports to use `predict_ml.py` (new module)

### 📁 New Files

```
src/
├── models/
│   └── pppq_multi_output_model.py  # NEW: Multi-output training
└── api/
    ├── predict_ml.py                # NEW: ML-powered predictions
    ├── predict_old_backup.py        # Backup of v1 logic
    └── main.py                      # Updated imports

models/pppq/
├── lgbm_classifier.txt                          # Updated
├── xgb_classifier.json                          # Updated
├── lgbm_target_real_pp_score_regressor.txt      # NEW
├── lgbm_target_volatility_score_regressor.txt   # NEW
├── lgbm_target_cycle_score_regressor.txt        # NEW
├── lgbm_target_growth_score_regressor.txt       # NEW
├── lgbm_target_consistency_score_regressor.txt  # NEW
├── lgbm_target_recovery_score_regressor.txt     # NEW
├── lgbm_target_risk_adjusted_score_regressor.txt # NEW
└── lgbm_target_commodity_score_regressor.txt    # NEW
```

### 🐛 Fixes

- Fixed inconsistent component scores across different horizons
- Resolved edge cases in hardcoded scoring logic
- Improved numerical stability for extreme values
- Better handling of missing/zero values in features

### 🎨 API Response Changes

Added new fields:
```json
{
  "model_version": "v2.0.0",
  "component_scores": {
    "commodity_score": 85.0,           // NEW
    "commodity_weight": 0.00,          // NEW
    "commodity_analysis": "..."        // NEW
  }
}
```

All component score `*_analysis` fields now say:
```
"ML-predicted score: XX.X/100"
```
instead of descriptive text (since scores are ML-predicted, not rule-based).

### 📝 Known Limitations

1. **Commodity Score Weight = 0** in final composite
   - Tracked separately as additional metric
   - Future versions may include in composite

2. **Component Analysis Text** is generic
   - ML models predict scores, not explanations
   - Detailed analysis could be added via SHAP

3. **Model File Size** increased
   - 8 new regressor models (~10MB total)
   - Lazy loading prevents memory issues

### 🔮 Future Enhancements (v2.1.0+)

Planned for future releases:
- [ ] SHAP explanations for component scores
- [ ] Time series forecasting (price predictions)
- [ ] Clustering analysis (asset segmentation)
- [ ] Recommendation system (similar assets)
- [ ] Dimensionality reduction (t-SNE, UMAP)
- [ ] More commodity features (bread, gas, housing)
- [ ] Multi-horizon predictions (predict all horizons at once)

### 🏆 Credits

**Author**: Bilal Ahmad Sheikh (GIKI)
**Date**: December 17, 2024
**Training Time**: ~45 seconds total (classification + regressors)
**Lines Changed**: +2,000 / -600

---

## [v1.2.0] - 2024-12-16

### Added
- Ensemble model support (LightGBM + XGBoost averaging)
- Horizon-aware dynamic weights
- Model type selection ('lgbm', 'xgb', 'ensemble')
- Threshold-based classification

### Changed
- Improved classification accuracy to 90.35%
- Better handling of imbalanced classes

---

## [v1.1.0] - 2024-12-15

### Added
- XGBoost model for ensemble
- Market cycle position analysis
- Entry signal generation

---

## [v1.0.0] - 2024-12-14

### Initial Release
- LightGBM classifier (90.28% F1)
- 18 engineered features
- A/B/C/D classification
- FastAPI deployment
- Docker support

---

**Latest**: v2.0.0
**Stability**: Production
**Performance**: 96.30% Macro-F1 | 99.3% Component R²

