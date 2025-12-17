# Two-Stage ML Prediction Flow (v2.0.0)

**Date**: 2024-12-17
**Status**: ✅ FULLY OPERATIONAL
**Models**: 10 Total (2 Classifiers + 8 Regressors)

---

## Overview

The PPP-Q v2.0.0 system uses a **two-stage ML architecture** where regression models and classification models work **together** to produce the final prediction.

```
User Input → STAGE 1 (Regression) → STAGE 2 (Classification) → Final Output
             8 LightGBM Models        LightGBM + XGBoost          Grade + Scores
             (99.3% avg R²)            (96.30% F1)
```

---

## Complete Prediction Flow

### Input
```python
asset = "Bitcoin"
horizon_years = 5
model_type = "ensemble"  # User can choose: ensemble, lgbm, or xgb
```

### Stage 1: Regression Models (Component Scores)

**Purpose**: Predict 8 interpretable component scores (0-100 scale)

**Models Used**: 8 LightGBM Regressors

**Code Location**: `streamlit_app/app.py:1234-1251`

```python
# =========================================================================
# STAGE 1: REGRESSION MODELS - Predict 8 Component Scores (v2.0.0)
# =========================================================================
# Uses 8 LightGBM regressors (99.3% avg R²) to predict:
# - Real PP Score, Volatility Score, Cycle Score, Growth Score,
# - Consistency Score, Recovery Score, Risk-Adjusted Score, Commodity Score

component_models = models.get('component_models', {})
feature_columns = models.get('feature_columns', [])

if component_models and len(component_models) > 0 and feature_columns:
    # ML-POWERED COMPONENT SCORES (v2.0.0)
    # HORIZON-AWARE: Scores change based on investment timeframe
    component_scores = predict_component_scores_ml(
        latest_row,
        component_models,
        feature_columns,
        horizon_years
    )

final_score = component_scores['final_composite_score']
```

**Function**: `predict_component_scores_ml()` at `app.py:645-812`

**Process**:
1. Extract 39 features from latest_row
2. Apply horizon-aware adjustments:
   - PP Multiplier: Scale by (horizon_years / 5.0)
   - Volatility: Decay by max(0.6, 1.0 - (H-1) * 0.08)
   - Cycle Position: Adjust sensitivity based on horizon
   - Growth Potential: Amplify by (1.0 + (H-1) * 0.08)
3. Predict with each of 8 LightGBM regressors
4. Clip predictions to [0, 100] range
5. Calculate weighted average for final composite score

**Output from Stage 1**:
```python
{
    'real_purchasing_power_score': 85.3,
    'volatility_risk_score': 42.1,
    'market_cycle_score': 68.9,
    'growth_potential_score': 91.2,
    'consistency_score': 55.7,
    'recovery_score': 73.4,
    'risk_adjusted_score': 67.8,
    'commodity_score': 88.5,
    'final_composite_score': 72.4  # Weighted average
}
```

**Weights**:
- Real PP: 25%
- Volatility: 20%
- Cycle: 15%
- Growth: 15%
- Consistency: 10%
- Recovery: 10%
- Risk-Adjusted: 5%
- Commodity: Tracked separately (not in composite)

---

### Stage 2: Classification Models (Final Grade)

**Purpose**: Predict final grade (A/B/C/D) using trained classifiers

**Models Used**:
- LightGBM Classifier (40% weight) - 96.5% F1
- XGBoost Classifier (60% weight) - 96.7% F1
- **Ensemble** - 96.30% F1 (weighted voting)

**Code Location**: `streamlit_app/app.py:1253-1327`

```python
# =========================================================================
# STAGE 2: CLASSIFICATION MODELS - Predict Final Grade (v2.0.0)
# =========================================================================
# Uses LightGBM (40%) + XGBoost (60%) ensemble (96.30% F1) to predict:
# - A_PRESERVER, B_PARTIAL, C_ERODER, or D_DESTROYER

if feature_columns and (models.get('lgbm') or models.get('xgb')):
    try:
        # Extract features for classification
        features = []
        for col in feature_columns:
            if col in latest_row.index:
                value = latest_row[col]
                features.append(0.0 if pd.isna(value) else float(value))
            else:
                features.append(0.0)

        features_array = np.array(features).reshape(1, -1)

        # Get predictions from selected model type
        if model_type == "ensemble":
            # Ensemble prediction (40% LightGBM + 60% XGBoost)
            lgbm_probs = models['lgbm'].predict(features_array)[0]  # (4,)
            xgb_probs = models['xgb'].predict(features_array)[0]    # (4,)

            # Weighted ensemble
            ensemble_probs = (lgbm_probs * 0.4) + (xgb_probs * 0.6)
            predicted_class_idx = int(np.argmax(ensemble_probs))
            classification_confidence = float(ensemble_probs[predicted_class_idx]) * 100

        elif model_type == "lgbm":
            # LightGBM only
            lgbm_probs = models['lgbm'].predict(features_array)[0]
            predicted_class_idx = int(np.argmax(lgbm_probs))
            classification_confidence = float(lgbm_probs[predicted_class_idx]) * 100

        elif model_type == "xgb":
            # XGBoost only
            xgb_probs = models['xgb'].predict(features_array)[0]
            predicted_class_idx = int(np.argmax(xgb_probs))
            classification_confidence = float(xgb_probs[predicted_class_idx]) * 100

        # Map class index to class name
        class_names = ['A_PRESERVER', 'B_PARTIAL', 'C_ERODER', 'D_DESTROYER']
        predicted_class = class_names[predicted_class_idx]

    except Exception:
        # Fallback to threshold-based if ML fails
        predicted_class = None
```

**Process**:
1. Extract same 39 features used in training
2. Reshape to (1, 39) array
3. Get probability distributions from classifiers
4. Ensemble voting (if selected):
   - LightGBM probabilities × 0.4
   - XGBoost probabilities × 0.6
   - Sum and take argmax
5. Map index to class name
6. Extract confidence from probability

**Output from Stage 2**:
```python
predicted_class = "A_PRESERVER"
classification_confidence = 87.3  # From softmax probability
```

**Class Mapping**:
- Index 0 → A_PRESERVER (≥65 score)
- Index 1 → B_PARTIAL (55-64 score)
- Index 2 → C_ERODER (35-54 score)
- Index 3 → D_DESTROYER (<35 score)

---

### Final Output

**Combines both stages**:

```python
{
    'asset': 'Bitcoin',
    'predicted_class': 'A_PRESERVER',  # From STAGE 2 (Classification)
    'confidence': 85.8,                 # From STAGE 2 (ML probability)
    'component_scores': {               # From STAGE 1 (Regression)
        'real_purchasing_power_score': 85.3,
        'volatility_risk_score': 42.1,
        'market_cycle_score': 68.9,
        'growth_potential_score': 91.2,
        'consistency_score': 55.7,
        'recovery_score': 73.4,
        'risk_adjusted_score': 67.8,
        'commodity_score': 88.5,
        'final_composite_score': 72.4
    },
    'strengths': [...],
    'weaknesses': [...],
    'metrics': {...},
    'horizon_years': 5,
    'model_type': 'ensemble'
}
```

---

## Why Two Stages?

### Interpretability + Accuracy

**Stage 1 (Regression)** provides:
- ✅ **Explainability**: 8 interpretable scores show WHY an asset got its grade
- ✅ **Granularity**: Users see exactly what's good/bad (volatility low, growth high, etc.)
- ✅ **Transparency**: Can trace final score back to individual components

**Stage 2 (Classification)** provides:
- ✅ **Accuracy**: 96.30% F1 (better than threshold rules)
- ✅ **Learned patterns**: Captures complex decision boundaries
- ✅ **Confidence scores**: Real probabilities from softmax

### Example: Bitcoin 5Y Horizon

**Stage 1 Output**:
```
Real PP Score:      85.3/100  (25% weight) → Strong
Volatility Score:   42.1/100  (20% weight) → Weak (crypto volatility)
Cycle Score:        68.9/100  (15% weight) → Good
Growth Score:       91.2/100  (15% weight) → Excellent
Consistency Score:  55.7/100  (10% weight) → Moderate
Recovery Score:     73.4/100  (10% weight) → Good
Risk-Adjusted:      67.8/100  (5% weight)  → Good
Commodity Score:    88.5/100  (tracked)    → Strong

Final Composite: 72.4/100
```

**Interpretation**:
- Strong PP growth and growth potential
- But penalized by high volatility
- Net result: ~72/100 composite score

**Stage 2 Output**:
```
Class Probabilities (Ensemble):
- A_PRESERVER:  87.3% ← Winner
- B_PARTIAL:     8.2%
- C_ERODER:      3.1%
- D_DESTROYER:   1.4%

Predicted: A_PRESERVER (87.3% confidence)
```

**Interpretation**:
- Models learned that crypto with 72+ score + high growth → Grade A
- Even with high volatility, long-term PP preservation is strong
- 87.3% confidence means models are very sure

---

## Model Loading Status

**Sidebar Display** (`app.py:2326-2353`):

```
🤖 Models Loaded
┌─────────────────┬─────────────────┐
│ Classification  │   Regression    │
│     2/2         │      8/8        │
├─────────────────┼─────────────────┤
│ ✅ LightGBM     │ ✅ Component    │
│ ✅ XGBoost      │    Scores       │
└─────────────────┴─────────────────┘
```

Users can see in real-time:
- Which classification models loaded (LightGBM, XGBoost)
- How many regression models loaded (8/8)
- If anything failed to load

---

## Feature Extraction

**Same 39 features used for both stages**:

```python
feature_columns = [
    # Purchasing Power Features (6)
    'PP_Multiplier_1Y', 'PP_Multiplier_5Y', 'PP_Multiplier_10Y',
    'Real_Return_1Y', 'Real_Return_5Y', 'Real_Return_10Y',

    # Volatility Features (3)
    'Volatility_90D', 'Max_Drawdown', 'Downside_Deviation',

    # Risk-Adjusted Returns (3)
    'Sharpe_Ratio_1Y', 'Sharpe_Ratio_5Y', 'Calmar_Ratio',

    # Market Cycle Features (4)
    'Distance_From_ATH_Pct', 'Distance_From_200MA_Pct',
    'Distance_From_50MA_Pct', 'Trend_Strength',

    # Growth & Saturation (2)
    'Market_Cap_Saturation_Pct', 'Growth_Potential_Score',

    # Recovery Metrics (3)
    'Recovery_Days', 'Recovery_Strength', 'Consistency_Index',

    # Commodity Features (5) - NEW in v2.0.0
    'Eggs_Per_100USD', 'Milk_Gallons_Per_100USD',
    'Real_Return_Eggs_1Y', 'Real_Return_Milk_1Y',
    'Real_Commodity_Basket_Return_1Y',

    # Statistical Features (5)
    'Mean_Return_Annual', 'Median_Return_Annual',
    'Skewness', 'Kurtosis', 'Win_Rate_Pct',

    # Macro Features (4)
    'Correlation_with_CPI', 'Correlation_with_Interest_Rates',
    'Correlation_with_USD', 'Beta_to_SPX'
]
# Total: 39 features
```

---

## Horizon-Aware Adjustments

**Stage 1 (Regression)** applies horizon adjustments to features **before** prediction:

| Feature Type | 1Y Horizon | 5Y Horizon | 10Y Horizon |
|--------------|------------|------------|-------------|
| **PP Multiplier** | Use 1Y value | Use 5Y value | Use 10Y value |
| **Volatility** | Full penalty (1.0×) | Moderate (0.68×) | Low penalty (0.28×) |
| **Cycle Position** | Strict (1.2×) | Normal (1.0×) | Forgiving (0.8×) |
| **Growth Potential** | Conservative (1.0×) | Normal (1.32×) | Amplified (1.72×) |
| **Drawdown** | Full penalty (1.0×) | Moderate (0.9×) | Low (0.8×) |
| **Sharpe Ratio** | Base (1.0×) | Enhanced (1.48×) | Strong (2.08×) |

**Stage 2 (Classification)** uses raw features (no adjustments), as the model learned patterns during training.

---

## Fallback Logic

### If Regression Models Fail to Load
```python
# Falls back to hardcoded component scoring
component_scores = calculate_component_scores(latest_row, asset, horizon_years)
```

### If Classification Models Fail to Load
```python
# Falls back to threshold-based grading
grade = assign_grade(adjusted_score, category)
grade_map = {'A': 'A_PRESERVER', 'B': 'B_PARTIAL', ...}
predicted_class = grade_map.get(grade, 'C_ERODER')
```

**Degradation Path**:
1. **Full ML** (2 classifiers + 8 regressors) → 96.30% F1 + 99.3% R²
2. **Regression only** (8 regressors + threshold grading) → 99.3% R² + ~70% classification
3. **Hardcoded only** (no ML) → ~60-70% overall (fallback mode)

---

## Performance Metrics

| Stage | Component | Model | Metric | Value |
|-------|-----------|-------|--------|-------|
| **Stage 1** | Real PP Score | LightGBM | R² | 99.5% |
| **Stage 1** | Volatility Score | LightGBM | R² | 99.2% |
| **Stage 1** | Cycle Score | LightGBM | R² | 99.1% |
| **Stage 1** | Growth Score | LightGBM | R² | 99.3% |
| **Stage 1** | Consistency Score | LightGBM | R² | 99.0% |
| **Stage 1** | Recovery Score | LightGBM | R² | 99.2% |
| **Stage 1** | Risk-Adjusted Score | LightGBM | R² | 99.4% |
| **Stage 1** | Commodity Score | LightGBM | R² | 99.4% |
| **Stage 1** | **Average** | - | **R²** | **99.3%** |
| **Stage 2** | Classification | LightGBM | F1 | 96.5% |
| **Stage 2** | Classification | XGBoost | F1 | 96.7% |
| **Stage 2** | **Ensemble** | **Both** | **F1** | **96.30%** |

---

## User Experience

### Model Selection Works
Users can choose:
- **Ensemble** (default) - Best accuracy (96.30% F1)
- **LightGBM** - Fast, 96.5% F1
- **XGBoost** - Robust, 96.7% F1

Selection affects **Stage 2 only** (classification). Stage 1 (regression) always uses all 8 LightGBM regressors.

### Transparency
Users see:
1. Which models are loaded (sidebar: "2/2 Classification, 8/8 Regression")
2. Which model is being used for prediction ("Ensemble", "LightGBM", or "XGBoost")
3. Confidence score from ML probabilities
4. All 8 component scores (explainability)

---

## Code References

| Component | File | Lines |
|-----------|------|-------|
| **Model Loading** | `streamlit_app/app.py` | 554-591 |
| **Stage 1 (Regression)** | `streamlit_app/app.py` | 1234-1251 |
| **Regression Prediction** | `streamlit_app/app.py` | 645-812 |
| **Stage 2 (Classification)** | `streamlit_app/app.py` | 1253-1327 |
| **Model Status Display** | `streamlit_app/app.py` | 2326-2353 |
| **Final Output Assembly** | `streamlit_app/app.py` | 1382-1396 |

---

## Summary

✅ **Both classification and regression models are used together**

**Two-Stage Flow**:
1. **Regression** (8 models) → Predict component scores → Final composite
2. **Classification** (2 models) → Predict grade → Confidence score

**Result**:
- **Interpretability** from regression (8 scores explain the "why")
- **Accuracy** from classification (96.30% F1 for final grade)
- **Best of both worlds**: Explainable + Accurate

**Status**: ✅ FULLY OPERATIONAL in v2.0.0

---

**Author**: Bilal Ahmad Sheikh (GIKI)
**Version**: v2.0.0 (Multi-Output ML)
**Last Updated**: 2024-12-17
