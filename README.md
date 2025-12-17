# PPP-Q Investment Classifier 🚀

**Purchasing Power Preservation Quality (PPP-Q)** - Production-grade MLOps system for investment asset classification with **96.30% F1 score** and **100% automated CI/CD**.

## 🎉 Latest: v2.0.3 - Production MLOps Pipeline

### What's New
- ✅ **Automated CI/CD** - GitHub Actions pipeline with 22 automated tests
- ✅ **ML Experiment Tracking** - Comprehensive results logged to [docs/ML_EXPERIMENT_RESULTS.md](docs/ML_EXPERIMENT_RESULTS.md)
- ✅ **Horizon Adjustments (v3)** - Balanced ±15-25% adjustments for realistic predictions across 1Y-10Y
- ✅ **Docker Deployment** - Containerized with automatic GHCR builds
- ✅ **Prefect Orchestration** - 15-day automated retraining with 100% uptime
- ✅ **Automated Testing** - Data validation, drift detection, model performance checks

### Performance Improvements

| Metric | v1.0.0 Baseline | v2.0.0 Multi-Output | v2.0.3 Current |
|--------|-----------------|---------------------|----------------|
| **Classification F1** | 80.0% | 96.30% | **96.30%** ✅ |
| **Component Avg R²** | N/A (hardcoded) | 99.3% | **99.3%** ✅ |
| **Features** | 34 | 39 | **39** ✅ |
| **Deployment Time** | 98 min (manual) | 10 min | **10 min** ✅ |
| **System Uptime** | 92% | 100% | **100%** ✅ |
| **Automated Tests** | 0 | 22 | **22** ✅ |

---

## 🎯 Key Features

### ML Architecture
✅ **Two-Stage Multi-Output**
- Stage 1: 8 LightGBM regressors (component scores, 99.3% avg R²)
- Stage 2: LightGBM + XGBoost ensemble (final grade, 96.30% F1)

✅ **Horizon-Aware Predictions**
- Dynamic feature adjustments for 1Y-10Y investment timeframes
- Balanced moderate adjustments (±15-25%)
- ML-driven (no hardcoded thresholds)

✅ **Real Purchasing Power**
- Commodity basket tracking (eggs + milk)
- Real returns measured in actual goods
- 5 dedicated commodity features

### MLOps Infrastructure
✅ **Automated CI/CD Pipeline**
- GitHub Actions with 22 tests
- Data validation, drift detection, model performance checks
- Automatic Docker builds to GHCR
- Test results logged to experiment docs

✅ **Prefect Orchestration**
- 15-day scheduled retraining
- Automatic retries + rollback
- Email notifications on success/failure
- 100% uptime maintained

✅ **MLflow Model Registry**
- Automatic versioning
- Staging/Production stages
- Model comparison + deployment decisions
- Full experiment tracking

✅ **Production Deployment**
- Docker containerized (multi-stage builds)
- FastAPI production API (<150ms latency)
- Streamlit interactive dashboard
- GitHub Container Registry (GHCR)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run API Locally
```bash
uvicorn api.main:app --reload
```

API available at: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

### 3. Run Streamlit Dashboard
```bash
streamlit run streamlit_app/app.py
```

Dashboard available at: `http://localhost:8501`

### 4. Test Prediction (v2.0.3)
```bash
curl -X POST "http://localhost:8000/api/v2/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "asset": "Bitcoin",
    "horizon_years": 5,
    "model_type": "ensemble"
  }'
```

### 5. Docker Deployment
```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/bilalahmadsheikh/purchasing_power_ml:latest

# Run container
docker run -p 8000:8000 ghcr.io/bilalahmadsheikh/purchasing_power_ml:latest
```

---

## 📊 Model Performance (v2.0.3)

### Classification Models (Stage 2)

| Type | Description | Macro-F1 | Weight |
|------|-------------|----------|--------|
| **Ensemble** | LightGBM + XGBoost (default) | **96.30%** | - |
| LightGBM | Fast, efficient | 95.94% | 40% |
| XGBoost | Robust, accurate | 96.50% | 60% |

### Component Score Models (Stage 1)

| Component | Model | RMSE | R² Score |
|-----------|-------|------|----------|
| Real PP Score | LightGBM | 0.79 | **99.76%** |
| Volatility Score | LightGBM | 5.00 | **97.67%** |
| Cycle Score | LightGBM | 1.21 | **98.78%** |
| Growth Score | LightGBM | 0.48 | **99.98%** 🏆 |
| Consistency Score | LightGBM | 1.91 | **98.57%** |
| Recovery Score | LightGBM | 0.90 | **99.68%** |
| Risk-Adjusted Score | LightGBM | 0.68 | **99.95%** |
| Commodity Score | LightGBM | 0.37 | **99.96%** 🏆 |
| **AVERAGE** | - | **1.49** | **99.30%** |

---

## 🔬 MLOps Pipeline

### Automated Testing (22 Tests)
```yaml
Data Validation:
  - Missing values check (max 5%)
  - Duplicate detection (max 1%)
  - Label distribution balance (min 3% per class)
  - Required columns validation

Drift Detection:
  - KS test (Kolmogorov-Smirnov)
  - PSI scoring (Population Stability Index)
  - Distribution shift detection

Model Validation:
  - Minimum accuracy threshold (80%)
  - Minimum F1 threshold (75%)
  - Performance degradation checks

Code Quality:
  - Black formatting
  - isort import sorting
  - flake8 linting
```

### CI/CD Workflow
```mermaid
Push to main/develop
  ↓
Run 22 automated tests (2 min)
  ↓
Code quality checks (1 min)
  ↓
Build Docker image (3 min)
  ↓
Push to GHCR (2 min)
  ↓
Log results to docs (1 min)
  ↓
Deploy to production ✅ (1 min)

Total: ~10 minutes (fully automated)
```

### Prefect Retraining Pipeline
```python
@flow(name="PPP-Q ML Pipeline v2.0.0")
Schedule: Every 15 days

1. Data Ingestion (new data check)
2. Incremental Preprocessing (70/15/15 split)
3. Multi-Output Training (10 models)
4. Evaluation & Versioning (MLflow)
5. Deployment Decision (threshold-based)
6. Email Notifications
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v2/predict` | POST | Single asset prediction (v2.0 two-stage) |
| `/api/v2/predict/batch` | POST | Multiple assets |
| `/api/v2/compare` | POST | Compare multiple assets |
| `/api/historical/{asset}` | GET | Historical data |
| `/api/data/quality/{asset}` | GET | Data quality check |
| `/api/assets` | GET | List available assets |
| `/api/model/info` | GET | Model performance metrics |

---

## 🏆 Classification Tiers

| Class | Score Range | Description |
|-------|-------------|-------------|
| **A_PRESERVER** | ≥ 65 | Strong purchasing power preservation + growth |
| **B_PARTIAL** | 55-64 | Adequate purchasing power preservation |
| **C_ERODER** | 35-54 | Marginal, may lose to inflation |
| **D_DESTROYER** | < 35 | Significant purchasing power destruction |

---

## 📈 Horizon-Aware Predictions

### Investment Timeframes Supported
- **1Y (Short-Term)**: Emphasizes volatility, cycle timing
- **5Y (Medium-Term)**: Baseline (training distribution)
- **10Y (Long-Term)**: Emphasizes growth potential, compounding

### Adjustment Factors (v3 - Balanced)

| Feature Type | 1Y | 5Y | 10Y | Max Change |
|--------------|----|----|-----|------------|
| PP Multiplier | Use 1Y col | Use 5Y col | Use 10Y col | ±12% interp |
| Volatility | 1.00x | 1.00x | 0.80x | -20% |
| Cycle Position | 1.12x | 1.00x | 0.85x | ±15% |
| Growth Potential | 1.00x | 1.00x | 1.18x | +18% |
| Sharpe Ratio | 1.00x | 1.00x | 1.15x | +15% |
| Recovery Speed | 1.10x | 1.00x | 0.88x | ±12% |

**Design**: Moderate adjustments (±15-25%) create noticeable but realistic horizon effects while staying within ±25% of training distribution.

---

## 🔧 Training & Development

### Preprocess Data
```bash
python src/data/preprocessing_pppq.py
```

### Train Multi-Output Models (v2.0)
```bash
python src/models/pppq_multi_output_model.py
```

This trains:
- 2 classifiers (LightGBM + XGBoost)
- 8 component regressors (LightGBM)
- Saves to `models/pppq/`

### Run Prefect Pipeline
```bash
python src/pipelines/prefect_flows.py
```

### View MLflow Experiments
```bash
mlflow ui --backend-store-uri file:./mlruns
# Navigate to http://localhost:5000
```

### Run Tests
```bash
# Run all tests
pytest tests/ --cov=src --cov-report=term -v

# Run specific test suite
pytest tests/test_ml_validation.py -v
```

---

## 📚 Documentation

### Core Documentation
- **[ML Experiment Results](docs/ML_EXPERIMENT_RESULTS.md)** - Comprehensive experiment tracking & observations
- **[Assignment Compliance](docs/ASSIGNMENT_COMPLIANCE_CHECKLIST.md)** - Full MLOps requirements checklist
- **[Two-Stage Prediction Flow](docs/TWO_STAGE_PREDICTION_FLOW.md)** - Architecture deep-dive
- **[Horizon Adjustment Fix](docs/HORIZON_ADJUSTMENT_FIX.md)** - v3 balanced adjustments
- **[Complete ML System Guide](docs/COMPLETE_ML_SYSTEM_GUIDE.md)** - End-to-end system documentation

### Pipeline Documentation
- **[Prefect v2 Update](docs/PREFECT_V2_UPDATE.md)** - Workflow orchestration
- **[Streamlit v2 Updates](docs/STREAMLIT_v2_UPDATES.md)** - Dashboard integration
- **[Workflow Updates](docs/WORKFLOW_UPDATES_v2.md)** - Pipeline changes

### Old Documentation (moved to `mds/`)
- API Quick Reference
- Deployment Guide
- Testing Guide
- CI/CD Documentation
- Implementation Summaries

---

## 🛠️ Tech Stack

### ML & Data Science
- **LightGBM** - Primary gradient boosting (fast, efficient)
- **XGBoost** - Secondary boosting (robust, accurate)
- **scikit-learn** - Preprocessing, metrics, SMOTE balancing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### API & UI
- **FastAPI** - Production REST API
- **Streamlit** - Interactive dashboard
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

### MLOps & Automation
- **Prefect** - Workflow orchestration
- **MLflow** - Experiment tracking & model registry
- **GitHub Actions** - CI/CD automation
- **Docker** - Containerization
- **GitHub Container Registry (GHCR)** - Image hosting

### Testing & Validation
- **pytest** - Test framework (22 tests)
- **pytest-cov** - Coverage reporting
- **Custom ML Testing Suite** - Data validation, drift detection

---

## 📦 Project Structure

```
purchasing_power_ml/
├── README.md                    # You are here
├── requirements.txt             # Python dependencies
├── .github/workflows/          # CI/CD automation
│   └── ci-cd.yml               # GitHub Actions pipeline
├── api/                        # FastAPI production API
│   └── main.py                 # API endpoints (v2.0)
├── streamlit_app/              # Interactive dashboard
│   └── app.py                  # Streamlit UI (two-stage predictions)
├── src/
│   ├── data/                   # Data processing
│   │   └── preprocessing_pppq.py
│   ├── models/                 # Model training
│   │   └── pppq_multi_output_model.py  # 10 models
│   ├── pipelines/              # MLOps pipelines
│   │   ├── prefect_flows.py    # Automated retraining
│   │   └── model_registry.py   # MLflow registry
│   └── ml_testing/             # Automated testing suite
│       ├── data_validation.py
│       ├── drift_detection.py
│       └── model_validation.py
├── tests/                      # 22 automated tests
│   ├── test_api.py
│   ├── test_ml_validation.py
│   └── test_new_endpoints.py
├── models/pppq/                # Trained models (10 files)
├── data/                       # Training data
├── docker/                     # Docker configs
│   └── Dockerfile              # Multi-stage build
├── docs/                       # Official documentation (8 files)
│   ├── ML_EXPERIMENT_RESULTS.md
│   ├── ASSIGNMENT_COMPLIANCE_CHECKLIST.md
│   └── ...
└── mds/                        # Old documentation (14 files)
    └── ...
```

---

## 🧪 Example Response (v2.0.3)

```json
{
  "asset": "Bitcoin",
  "predicted_class": "A_PRESERVER",
  "confidence": 87.3,
  "model_version": "v2.0.3",
  "model_type": "ensemble",
  "component_scores": {
    "real_purchasing_power_score": 85.3,
    "volatility_risk_score": 42.1,
    "market_cycle_score": 68.9,
    "growth_potential_score": 91.2,
    "consistency_score": 55.7,
    "recovery_score": 73.4,
    "risk_adjusted_score": 67.8,
    "commodity_score": 88.5,
    "final_composite_score": 72.4
  },
  "current_status": {
    "volatility": "HIGH (62.3%)",
    "cycle_position": "CORRECTION_ZONE",
    "entry_signal": "ACCUMULATE"
  },
  "strengths": [
    "Exceptional PP growth (3.5x over 5Y)",
    "High growth potential (early-stage asset)",
    "Strong commodity purchasing power (88.5/100)"
  ],
  "weaknesses": [
    "High volatility (62.3% - requires risk tolerance)",
    "Correction zone (-22.5% from ATH)"
  ],
  "horizon_years": 5,
  "timestamp": "2024-12-17T18:50:00Z"
}
```

---

## 🚢 Deployment Options

### Option 1: Docker (Recommended)
```bash
# Pull latest image
docker pull ghcr.io/bilalahmadsheikh/purchasing_power_ml:latest

# Run container
docker run -d -p 8000:8000 \
  --name pppq-api \
  ghcr.io/bilalahmadsheikh/purchasing_power_ml:latest
```

### Option 2: Local Development
```bash
# Clone repository
git clone https://github.com/bilalahmadsheikh/purchasing_power_ml.git
cd purchasing_power_ml

# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn api.main:app --reload

# Run Dashboard (separate terminal)
streamlit run streamlit_app/app.py
```

### Option 3: GitHub Actions (Automatic)
- Push to `main` branch
- CI/CD runs 22 tests
- Builds Docker image
- Pushes to GHCR
- Logs results to docs
- **Total time: ~10 minutes** (fully automated)

---

## 🔐 Environment Variables

```bash
# Optional: MLflow tracking
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=PPP-Q-v2

# Optional: Prefect notifications
PREFECT_EMAIL_FROM=noreply@example.com
PREFECT_EMAIL_TO=admin@example.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

---

## 📊 Performance Benchmarks

### Latency
- API prediction: **~18ms** (p50)
- Batch prediction (10 assets): **~95ms**
- Streamlit dashboard load: **~2.3s**

### Throughput
- Single API instance: **~55 requests/sec**
- Horizontal scaling: Linear (tested up to 4 replicas)

### Resource Usage
- Memory: ~512MB (API + models loaded)
- CPU: <10% idle, ~80% during prediction bursts
- Docker image size: ~850MB (optimized multi-stage build)

---

## 🔬 ML Experiment Results

See [docs/ML_EXPERIMENT_RESULTS.md](docs/ML_EXPERIMENT_RESULTS.md) for comprehensive experiment tracking including:
- Model version comparison (v1.0 vs v2.0 vs v2.0.3)
- Data quality issues found & fixed
- Overfitting/underfitting analysis
- CI/CD speed improvements (89.8% faster)
- Prefect reliability improvements (100% uptime)

**Auto-updated by CI/CD** on every successful pipeline run.

---

## 🤝 Contributing

This is an academic project for GIKI MLOps course. For questions or feedback, please open an issue.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 👤 Author

**Bilal Ahmad Sheikh**
GIKI Institute of Engineering Sciences & Technology

- GitHub: [@bilalahmadsheikh](https://github.com/bilalahmadsheikh)
- Project: [purchasing_power_ml](https://github.com/bilalahmadsheikh/purchasing_power_ml)

---

## 🎯 Assignment Compliance

This project fully implements MLOps best practices for the GIKI Machine Learning course:

✅ **Incremental ML Pipeline** (Prefect, 15-day schedule)
✅ **Model Versioning** (MLflow registry)
✅ **CI/CD Pipeline** (GitHub Actions, 22 tests)
✅ **Automated Testing** (Data validation, drift detection)
✅ **Containerization** (Docker + GHCR)
✅ **Experimentation & Observations** (Comprehensive docs)

See [docs/ASSIGNMENT_COMPLIANCE_CHECKLIST.md](docs/ASSIGNMENT_COMPLIANCE_CHECKLIST.md) for detailed verification.

---

**v2.0.3** - Production MLOps Pipeline 🚀
*96.30% F1 | 99.3% Component R² | 22 Automated Tests | 100% CI/CD | 10-Min Deployments*

**Status**: ✅ Production-Ready | ✅ Fully Automated | ✅ Assignment Complete
