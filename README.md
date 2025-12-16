# PPP-Q Investment Classifier 🚀

**Purchasing Power Preservation Quality (PPP-Q)** ML system classifying investment assets based on purchasing power preservation with **Ensemble models** achieving **90.35% Macro-F1 score**.

## v1.2.0 New Features ✨

- **Ensemble Model** - Combines LightGBM + XGBoost for robust predictions
- **Dynamic Weights** - Component weights adjust by investment horizon
- **Model Type Selection** - Choose `lgbm`, `xgb`, or `ensemble` (default)
- **Threshold-Based Classification** - Scores map to A/B/C/D classes deterministically

## Features

✅ Multi-asset classification (A/B/C/D quality tiers)  
✅ Real-time predictions via FastAPI (150ms latency)  
✅ Ensemble model support (LightGBM + XGBoost)  
✅ Dynamic horizon-aware scoring  
✅ Actionable insights (entry signals, strengths/weaknesses)  
✅ Docker deployment  
✅ Production-ready with 17 engineered features  

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

### 3. Test Prediction (with Ensemble)
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
| `/model/info` | GET | Model performance |

## Model Types

| Type | Description | F1 Score |
|------|-------------|----------|
| `ensemble` | LightGBM + XGBoost average (default) | ~90.35% |
| `lgbm` | LightGBM only (fastest) | 90.28% |
| `xgb` | XGBoost only | 89.44% |

## Classification Thresholds

| Class | Score | Description |
|-------|-------|-------------|
| **A_PRESERVER** | ≥ 65 | Strong PP preservation + growth |
| **B_PARTIAL** | 55-64 | Adequate PP preservation |
| **C_ERODER** | 42-54 | Marginal, may lose to inflation |
| **D_DESTROYER** | < 42 | Significant PP destruction |

## Model Performance

| Metric | Score |
|--------|-------|
| **Macro-F1** | **90.35%** |
| Accuracy | 90.49% |
| Balanced Acc | 89.66% |

## Example Response
```json
{
  "asset": "Bitcoin",
  "predicted_class": "A_PRESERVER",
  "confidence": 99.98,
  "component_scores": {
    "real_purchasing_power_score": 95.0,
    "real_purchasing_power_weight": 0.20,
    "volatility_risk_score": 50.0,
    "volatility_risk_weight": 0.15,
    "final_composite_score": 79.5
  },
  "current_status": {
    "volatility": "MEDIUM (35.3%)",
    "cycle_position": "VALUE_ZONE",
    "entry_signal": "CONSIDER"
  },
  "strengths": [
    "Excellent PP preservation (4.27x over 5Y)",
    "Early-stage market - high growth potential"
  ],
  "weaknesses": [
    "Poor risk-adjusted performance (Sharpe: 0.25)"
  ]
}
```

## Documentation

- [API Quick Reference](API_QUICK_REFERENCE.md)
- [Model Changelog](MODEL_CHANGELOG.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Incremental Pipeline](INCREMENTAL_PIPELINE.md)
- [Testing Guide](TESTING.md)
- [CI/CD Documentation](CI_CD.md)

## Author

**Bilal Ahmad Sheikh**  
GIKI  
[GitHub](https://github.com/bilalahmadsheikh) | [LinkedIn](#)