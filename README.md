# PPP-Q Investment Classifier 🚀

**Purchasing Power Preservation Quality (PPP-Q)** ML system classifying investment assets based on purchasing power preservation with **Ensemble models** achieving **96.30% Macro-F1 score**.

## 🎉 v2.0.0 - Major Update: ML-Powered Component Scores

### What's New
- **ML-Predicted Component Scores** - All 8 component scores now predicted by ML models (R² = 99.3%!)
- **Egg/Milk Commodity Features** - Real purchasing power measured in actual goods (5 new features)
- **96.30% Macro-F1** - Up from 90.35% (6% improvement!)
- **39 Input Features** - Up from 18 (egg/milk purchasing power added)
- **8 Component Regressors** - Dedicated ML models for each score component
- **No Hardcoded Logic** - Pure ML predictions (no if/else scoring rules)

### Performance Improvements

| Metric | v1.2.0 | v2.0.0 | Improvement |
|--------|--------|--------|-------------|
| **Classification Macro-F1** | 90.35% | **96.30%** | +5.95% |
| **Component Score Accuracy** | N/A (hardcoded) | **R² = 99.3%** | ✨ NEW |
| **Features** | 18 | 39 | +21 features |
| **Commodity Features** | ❌ | ✅ Eggs/Milk | NEW |

## Features

✅ **Multi-asset classification** (A/B/C/D quality tiers)
✅ **Real-time predictions** via FastAPI (<150ms latency)
✅ **Ensemble model** (LightGBM + XGBoost)
✅ **Dynamic horizon-aware** scoring (1Y-10Y)
✅ **ML-predicted components** (8 regression models)
✅ **Real commodity purchasing power** (eggs/milk tracking)
✅ **Actionable insights** (entry signals, strengths/weaknesses)
✅ **Docker deployment**
✅ **Production-ready** with 39 engineered features

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run API Locally
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

API available at: `http://localhost:8001`
Interactive docs: `http://localhost:8001/docs`

### 3. Test Prediction (with v2.0.0 ML Scores)
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "asset": "Bitcoin",
    "horizon_years": 5,
    "model_type": "ensemble"
  }'
```

### 4. Docker Deployment
```bash
docker-compose -f docker/docker-compose.prod.yml up
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Single asset prediction (supports model_type) |
| `/predict/batch` | POST | Multiple assets |
| `/predict/upload` | POST | CSV file upload |
| `/compare` | POST | Compare multiple assets |
| `/asset/historical/{asset}` | GET | Historical data |
| `/data/quality/{asset}` | GET | Data quality check |
| `/assets` | GET | List available assets |
| `/model/info` | GET | Model performance (v2.0.0 metrics) |

## Model Performance (v2.0.0)

### Classification Models

| Type | Description | Macro-F1 |
|------|-------------|----------|
| **Ensemble** | LightGBM + XGBoost (default) | **96.30%** |
| `lgbm` | LightGBM only (fastest) | 95.94% |
| `xgb` | XGBoost only | 96.50% |

### Component Score Models (NEW in v2.0.0)

| Component | RMSE | R² Score |
|-----------|------|----------|
| **Real PP Score** (egg/milk included) | 0.791 | **0.998** |
| **Volatility Score** | 4.996 | **0.977** |
| **Cycle Score** | 1.211 | **0.988** |
| **Growth Score** | 0.478 | **1.000** |
| **Consistency Score** | 1.905 | **0.986** |
| **Recovery Score** | 0.901 | **0.997** |
| **Risk-Adjusted Score** | 0.684 | **0.999** |
| **Commodity Score** (🆕 eggs/milk) | 0.367 | **1.000** |
| **Average** | 1.417 | **0.993** |

## Classification Thresholds

| Class | Score | Description |
|-------|-------|-------------|
| **A_PRESERVER** | ≥ 65 | Strong PP preservation + growth |
| **B_PARTIAL** | 55-64 | Adequate PP preservation |
| **C_ERODER** | 42-54 | Marginal, may lose to inflation |
| **D_DESTROYER** | < 42 | Significant PP destruction |

## Example Response (v2.0.0)

```json
{
  "asset": "Bitcoin",
  "predicted_class": "A_PRESERVER",
  "confidence": 86.5,
  "model_version": "v2.0.0",
  "component_scores": {
    "real_purchasing_power_score": 95.0,
    "real_purchasing_power_weight": 0.25,
    "volatility_risk_score": 75.0,
    "volatility_risk_weight": 0.20,
    "market_cycle_score": 60.0,
    "market_cycle_weight": 0.15,
    "growth_potential_score": 80.0,
    "growth_potential_weight": 0.15,
    "consistency_score": 45.0,
    "consistency_weight": 0.10,
    "recovery_score": 28.0,
    "recovery_weight": 0.10,
    "risk_adjusted_score": 35.0,
    "risk_adjusted_weight": 0.05,
    "commodity_score": 85.0,
    "commodity_weight": 0.00,
    "commodity_analysis": "ML-predicted egg/milk purchasing power: 85.0/100",
    "final_composite_score": 69.2
  },
  "current_status": {
    "volatility": "MEDIUM (35.3%)",
    "cycle_position": "FAIR_VALUE",
    "entry_signal": "WATCH"
  },
  "strengths": [
    "Excellent PP preservation (4.27x over 5Y)",
    "Early-stage market - high growth potential",
    "Correction zone (-22.5%) - reasonable entry point"
  ],
  "weaknesses": [
    "Poor risk-adjusted performance (Sharpe: 0.25)",
    "Severe drawdown history (73.2%)"
  ],
  "investment_horizon_years": 5.0
}
```

## What Changed in v2.0.0?

### 🎯 Core Changes

1. **ML-Predicted Component Scores**
   - Removed ALL hardcoded scoring logic
   - 8 dedicated LightGBM regressors (one per component)
   - Average R² of 99.3% (nearly perfect predictions!)

2. **New Commodity Features**
   - `Eggs_Per_100USD` - How many eggs can $100 buy?
   - `Milk_Gallons_Per_100USD` - How many gallons can $100 buy?
   - `Real_Return_Eggs_1Y` - Asset return measured in eggs
   - `Real_Return_Milk_1Y` - Asset return measured in milk
   - `Real_Commodity_Basket_Return_1Y` - Combined egg/milk score

3. **Improved Classification**
   - Macro-F1: 90.35% → **96.30%** (+5.95%)
   - Better handling of edge cases
   - More robust to market volatility

### 🔧 Technical Improvements

- **Multi-Output Training** - Single training script for all 9 models
- **Horizon-Aware Features** - Dynamic adjustments for 1Y-10Y predictions
- **Better Model Management** - Singleton pattern for efficient loading
- **Backward Compatible** - API endpoints unchanged

### 📊 Training Data

- **Training Set**: 65,745 samples (2010-2021)
- **Validation Set**: 10,950 samples (2022-2023)
- **Test Set**: 10,725 samples (2024-2025)
- **Total Features**: 39 (up from 18)
- **Assets**: 15 (Bitcoin, Gold, SP500, etc.)

## Migration from v1.x to v2.0.0

No breaking changes! The API is backward compatible.

**What you get automatically:**
- ✅ Higher accuracy predictions
- ✅ ML-predicted component scores (no hardcoded logic)
- ✅ Commodity purchasing power tracking
- ✅ Better horizon-aware predictions

**What stays the same:**
- ✅ API endpoints
- ✅ Request/response format
- ✅ Docker deployment
- ✅ Classification thresholds (A/B/C/D)

## Training the Models

### Preprocess Data
```bash
python src/data/preprocessing_pppq.py
```

### Train Multi-Output Models (v2.0.0)
```bash
python src/models/pppq_multi_output_model.py
```

This trains:
- 1x LightGBM classifier (A/B/C/D)
- 1x XGBoost classifier (ensemble)
- 8x LightGBM regressors (component scores)

## Documentation

- [API Quick Reference](API_QUICK_REFERENCE.md)
- [Model Changelog](MODEL_CHANGELOG.md) - Full v2.0.0 release notes
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Incremental Pipeline](INCREMENTAL_PIPELINE.md)
- [Testing Guide](TESTING.md)
- [CI/CD Documentation](CI_CD.md)

## Tech Stack

- **ML**: LightGBM, XGBoost, scikit-learn
- **API**: FastAPI, Pydantic, Uvicorn
- **Data**: Pandas, NumPy
- **Deployment**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Monitoring**: MLflow (optional)

## License

MIT License

## Author

**Bilal Ahmad Sheikh**
GIKI
[GitHub](https://github.com/bilalahmadsheikh) | [LinkedIn](#)

---

**v2.0.0** - ML-Powered Component Scores + Egg/Milk Features 🥚🥛
*96.30% Macro-F1 | 99.3% Component R² | 39 Features | Production-Ready*
