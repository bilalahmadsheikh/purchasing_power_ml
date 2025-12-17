# Prefect Orchestration v2.0.0 Update

**Date**: 2024-12-17
**Status**: ✅ COMPLETED
**Version**: v2.0.0

---

## Summary

The Prefect ML pipeline orchestration has been successfully updated to support v2.0.0 multi-output model training with 10 models (2 classifiers + 8 component regressors).

---

## Changes Made

### 1. Updated Training Task

**File**: `src/pipelines/prefect_flows.py`

**OLD (v1.x)**:
```python
@task(name="train_models")
def train_models(train_df, val_df, test_df):
    """Train LightGBM, XGBoost, and Random Forest models"""
    # Trained only 3 classification models
    # No component score predictions
```

**NEW (v2.0.0)**:
```python
@task(name="train_multi_output_models")
def train_multi_output_models(train_df, val_df, test_df):
    """
    Train Multi-Output Models (v2.0.0)
    - 2 Classification Models (LightGBM + XGBoost)
    - 8 Component Regression Models (LightGBM)
    """
    # Runs pppq_multi_output_model.py via subprocess
    # Validates all 10 models created
    # Returns classification + component metrics
```

### 2. Model Validation

**New Validation Logic**:
```python
required_models = [
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

missing_models = [m for m in required_models if not (models_dir / m).exists()]
if missing_models:
    raise FileNotFoundError(f"Missing models: {missing_models}")
```

### 3. Metrics Tracking

**OLD Metrics**:
- LightGBM F1
- XGBoost F1
- Random Forest F1
- Ensemble F1

**NEW Metrics (v2.0.0)**:
- Classification F1 (macro)
- Classification Accuracy
- Component Avg R² (across 8 regressors)
- Component Min/Max R²
- Models Trained: 10

### 4. Model Registry Update

**File**: `src/pipelines/model_registry.py`

**Added Model Artifact Tracking**:
```python
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

### 5. Notification Updates

**NEW Notification Details**:
```python
details = {
    'New Data Rows': f"{new_rows:,}",
    'Model Version': 'v2.0.0 (Multi-Output)',
    'Models Trained': '10 (2 classifiers + 8 regressors)',
    'Classification F1': f"{classification_f1:.4f}",
    'Classification Accuracy': f"{accuracy:.4f}",
    'Component Avg R²': f"{avg_r2:.4f}",
    'Deployed': '✅ Yes' if deployed else '❌ No',
    'Run ID': run_id
}
```

### 6. Training Summary Update

**NEW Training Summary**:
```python
summary = {
    'training_date': datetime.now().isoformat(),
    'run_id': run_id,
    'version': '2.0.0',
    'best_model': 'Multi-Output-v2.0.0',
    'deployed': should_deploy,
    'classification_metrics': {...},
    'component_metrics': {...},
    'models_trained': {
        'total': 10,
        'classifiers': ['LightGBM', 'XGBoost'],
        'component_regressors': [
            'real_pp_score',
            'volatility_score',
            'cycle_score',
            'growth_score',
            'consistency_score',
            'recovery_score',
            'risk_adjusted_score',
            'commodity_score'
        ]
    }
}
```

---

## How to Use

### Manual Execution

```bash
# Run the Prefect pipeline manually
python -c "from src.pipelines.prefect_flows import run_pipeline; run_pipeline()"
```

### Scheduled Execution

```bash
# Run with Prefect scheduler (every 15 days)
python -c "from src.pipelines.prefect_flows import schedule_pipeline; schedule_pipeline()"
```

### Force Full Retrain

```python
from src.pipelines.prefect_flows import run_pipeline

# Force full retraining even if no new data
run_pipeline(force_full_retrain=True)
```

---

## Pipeline Flow (v2.0.0)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREFECT ML PIPELINE v2.0.0                   │
└─────────────────────────────────────────────────────────────────┘

1. ⬇️  FETCH NEW DATA (Incremental)
   └─→ Fetches only data since last run
   └─→ Appends to existing consolidated dataset
   └─→ If no new data, skips pipeline

2. 🔧 PREPROCESS DATA
   └─→ Processes new rows only
   └─→ Engineers 39 features (including egg/milk commodities)
   └─→ Creates time-based train/val/test splits

3. 🤖 TRAIN MULTI-OUTPUT MODELS (v2.0.0)
   └─→ Calls pppq_multi_output_model.py
   └─→ Trains 2 classifiers (LightGBM + XGBoost)
   └─→ Trains 8 component regressors (LightGBM)
   └─→ Validates all 10 models created

4. 📊 EVALUATE & VERSION
   └─→ Checks classification F1 vs production model
   └─→ Logs to model registry with v2.0.0 metadata
   └─→ Promotes to staging/production if improved

5. 📧 SEND NOTIFICATIONS
   └─→ Email notification with v2.0.0 metrics
   └─→ Includes classification + component scores
   └─→ Deployment status (✅ or ❌)
```

---

## Testing

### Test Prefect Pipeline

```bash
# Test data ingestion only
python -c "
from src.pipelines.prefect_flows import fetch_new_data
df, new_rows, df_new = fetch_new_data()
print(f'New rows: {new_rows}')
"

# Test preprocessing only
python -c "
from src.pipelines.prefect_flows import preprocess_data
import pandas as pd
df = pd.read_csv('data/raw/pppq/pppq_data.csv')
train, val, test = preprocess_data(df, 0, pd.DataFrame())
print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
"

# Test full pipeline
python src/pipelines/prefect_flows.py
```

---

## Model Registry

The model registry now tracks all 10 v2.0.0 models:

```python
# Get latest production model
from src.pipelines.model_registry import registry

production_model = registry.get_latest_model('Production')
print(f"Version: {production_model['params']['version']}")
print(f"Models: {production_model['params']['num_models']}")
print(f"Classification F1: {production_model['test_metrics']['macro_f1']}")
print(f"Component R²: {production_model['params']['component_avg_r2']}")

# Compare two models
comparison = registry.compare_models('20241215_143022', '20241216_091503')
print(comparison['improvements'])
```

---

## Deployment Decision Logic

```python
def should_deploy(new_metrics, threshold=0.001):
    """
    Deploy if:
    1. No production model exists (first deployment)
    2. New model F1 >= Production F1 + threshold (0.001)

    v2.0.0: Uses classification F1 as primary metric
    Component R² is tracked but not used for deployment decision
    """
    current_production = registry.get_latest_model('Production')

    if not current_production:
        return True  # First deployment

    prev_f1 = current_production['test_metrics']['macro_f1']
    new_f1 = new_metrics['macro_f1']

    return (new_f1 - prev_f1) >= threshold
```

---

## Performance Metrics (v2.0.0)

### Current Production Model

- **Classification F1**: 96.30% (macro)
- **Classification Accuracy**: 96.5%
- **Component Avg R²**: 99.3%
- **Models Trained**: 10
- **Training Time**: ~45 seconds (depending on hardware)

### Thresholds

- **Minimum F1 for Deployment**: Previous F1 + 0.001 (0.1% improvement)
- **Component R² Target**: ≥ 95% (currently 99.3%)

---

## Logs and Monitoring

### Log Files

```
logs/
├── pipeline.log           # Main pipeline logs
├── prefect_flows.log     # Prefect-specific logs
└── model_training.log    # Model training logs
```

### Reports

```
reports/
├── training_summary.json      # Latest training summary (v2.0.0)
├── feature_importance.csv     # Feature importances
└── model_registry.json        # All registered models
```

---

## Troubleshooting

### Issue: "Missing models after training"

**Solution**: Check that `pppq_multi_output_model.py` completed successfully:

```bash
python src/models/pppq_multi_output_model.py
# Should create all 10 model files in models/pppq/
```

### Issue: "No new data - pipeline skipped"

**Solution**: This is expected behavior. To force retraining:

```python
from src.pipelines.prefect_flows import run_pipeline
run_pipeline(force_full_retrain=True)
```

### Issue: "Model not deployed (insufficient improvement)"

**Solution**: This is expected if new model F1 ≤ production F1 + 0.001. You can:
1. Accept that current model is optimal
2. Lower threshold in `pipeline_config.py`
3. Collect more/better training data

---

## GitHub Actions Integration

The Prefect pipeline can also be triggered via GitHub Actions:

**Workflow**: `.github/workflows/automated-pipeline.yml`

**Triggers**:
- Weekly schedule (Monday 2 AM UTC)
- Manual dispatch
- Push to main/develop (data/models changed)

**Process**:
1. Runs data collection
2. Runs preprocessing
3. Calls Prefect `pppq_ml_pipeline()` flow
4. Commits and pushes new models (Git LFS)

---

## Summary

✅ **All Prefect orchestration updated for v2.0.0**

**Key Improvements**:
- Multi-output training (10 models instead of 3)
- Component score predictions (8 regressors)
- Horizon-aware feature engineering
- Commodity features (egg/milk)
- Improved metrics tracking
- Better model validation
- Enhanced notifications

**Status**: Production-ready

---

**Author**: Bilal Ahmad Sheikh (GIKI)
**Version**: v2.0.0
**Last Updated**: 2024-12-17
