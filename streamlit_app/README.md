# PPP-Q Investment Intelligence Dashboard

A comprehensive Streamlit dashboard for the Purchasing Power Preservation Quotient (PPP-Q) ML model.

## Features

### 🎯 Single Asset Analysis
- Select any asset from dropdown or type to search
- View detailed PPP-Q score and grade
- Component score breakdown with radar chart
- AI-generated insights (strengths, risks, recommendations)
- Score comparison across different investment horizons

### 📊 Asset Comparison
- Compare 2-6 assets side by side
- Visual bar chart comparison
- Detailed comparison table
- Best pick recommendation

### 📈 Correlations & Insights
- Asset category correlation heatmap
- Market regime indicators
- Diversification guidance

### 🤖 Model Selection
- **LightGBM** - Fast inference, 90.28% accuracy
- **XGBoost** - Robust predictions, 89.44% accuracy
- **Ensemble** - Combined approach, 90.35% accuracy

### 📅 Investment Horizon
- Slider or manual input (1-30 years)
- Dynamic weight adjustment based on horizon
- Short/Medium/Long-term context

## Deployment on Streamlit Cloud

### Step 1: Push to GitHub

The dashboard is already in your repository under `streamlit_app/`.

### Step 2: Connect Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `bilalahmadsheikh/purchasing_power_ml`
5. Branch: `main`
6. Main file path: `streamlit_app/app.py`
7. Click "Deploy"

### Step 3: Configure Secrets (Optional)

In Streamlit Cloud dashboard, add secrets:

```toml
[secrets]
API_BASE_URL = "https://your-api-url.com"
```

## Local Development

### Install Dependencies

```bash
cd streamlit_app
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

### Environment Variables

Create a `.env` file:

```
API_BASE_URL=http://localhost:8001
```

## API Integration

The dashboard connects to the FastAPI backend for predictions:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Get PPP-Q prediction |
| `/compare` | POST | Compare multiple assets |
| `/assets` | GET | List available assets |
| `/model/info` | GET | Model metadata |
| `/health` | GET | API health check |

## File Structure

```
streamlit_app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## Customization

### Change API URL

Set environment variable:
```bash
export API_BASE_URL="https://your-api.com"
```

Or modify in `app.py`:
```python
API_BASE_URL = "https://your-api.com"
```

### Add New Assets

Edit the `ASSETS` dictionary in `app.py`:
```python
ASSETS = {
    "NewAsset": {
        "category": "Category",
        "symbol": "SYM",
        "risk": "Medium",
        "liquidity": "High"
    },
    ...
}
```

## Version

- **Dashboard Version:** 1.0.0
- **Model Version:** v1.2.0
- **Last Updated:** December 2025
