# PPP-Q Investment Classifier 🚀

Purchasing Power Preservation Quality (PPP-Q) classifier using LightGBM with **95.4% Macro-F1 score**.

## Features

✅ Multi-asset classification (A/B/C/D quality tiers)  
✅ Real-time predictions via FastAPI  
✅ Market cycle awareness (ATH distance, Bitcoin halving)  
✅ Actionable insights (entry signals, strengths/weaknesses)  
✅ Docker deployment  
✅ Production-ready with 38 engineered features  

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run API Locally
```bash
cd C:\Users\bilaa\OneDrive\Desktop\purchasing_power_ml
uvicorn src.api.main:app --reload
```

API available at: `http://localhost:8000`

### 3. Test Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"asset": "Bitcoin", "horizon_years": 5}'
```

### 4. Docker Deployment
```bash
cd docker
docker-compose up --build
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Single asset prediction |
| `/predict/batch` | POST | Multiple assets |
| `/predict/upload` | POST | CSV file upload |
| `/assets` | GET | List available assets |
| `/model/info` | GET | Model performance |

## Model Performance

| Metric | Score |
|--------|-------|
| **Macro-F1** | **95.43%** |
| Accuracy | 94.42% |
| Balanced Acc | 94.80% |

## Example Response
```json
{
  "asset": "Bitcoin",
  "predicted_class": "B_PARTIAL",
  "confidence": 100.0,
  "current_status": {
    "volatility": "HIGH (42.3%)",
    "cycle_position": "VALUE_ZONE",
    "entry_signal": "CONSIDER"
  },
  "strengths": [
    "Strong PP growth (3.80x over 5Y)",
    "High growth potential"
  ],
  "weaknesses": [
    "Extreme volatility (42%)",
    "Severe drawdowns (-83%)"
  ]
}
```

## Author

**Bilal Ahmad Sheikh**  
GIKI
[GitHub](https://github.com/bilalahmadsheikh) | [LinkedIn](#)