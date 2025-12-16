
# PPP-Q ENHANCED CLASSIFIER - PRODUCTION TRAINING RESULTS

This report summarizes the training process and performance of the PPP-Q Enhanced Classifier, which utilizes LightGBM as the primary model, XGBoost for ensemble diversity, and Random Forest as a baseline. The training was conducted in December 2024 by Bilal Ahmad Sheikh from GIKI.

---

## 1. Data Loading & Validation

The datasets were successfully loaded:

*   **Train Set:** `(65745, 43)` samples/features
*   **Validation Set:** `(10950, 43)` samples/features
*   **Test Set:** `(10695, 43)` samples/features

The loaded data contains **38 features** and targets **4 classes**: `['A_PRESERVER', 'B_PARTIAL', 'C_ERODER', 'D_DESTROYER']`.

---

## 2. Feature Preparation

**Feature Columns:** 38 features were used for training.

**Class Mapping:**

*   `0: A_PRESERVER`
*   `1: B_PARTIAL`
*   `2: C_ERODER`
*   `3: D_DESTROYER`

**Class Distribution:**

| Class       | TRAIN Samples (Percentage) | VAL Samples (Percentage) | TEST Samples (Percentage) |
| :---------- | :------------------------- | :----------------------- | :------------------------ |
| A_PRESERVER | 21,366 (32.5%)             | 3,942 (36.0%)            | 4,332 (40.5%)             |
| B_PARTIAL   | 20,283 (30.9%)             | 5,550 (50.7%)            | 3,999 (37.4%)             |
| C_ERODER    | 19,834 (30.2%)             | 1,323 (12.1%)            | 1,495 (14.0%)             |
| D_DESTROYER | 4,262 (6.5%)               | 135 (1.2%)               | 869 (8.1%)                |

---

## 3. Model Training

### Primary Model: LightGBM

*   **Status:** Trained successfully.
*   **Best Iteration:** 189
*   **Training Time:** 30.55s
*   **Validation LogLoss at Best Iteration:** 0.138944

### Secondary Model: XGBoost

*   **Status:** Trained successfully.
*   **Best Iteration:** 153
*   **Training Time:** 12.50s

### Baseline Model: Random Forest

*   **Status:** Trained successfully.
*   **Training Time:** 48.47s

---

## 4. Model Evaluation (Test Set Results)

### LightGBM

| Metric            | Value  |
| :---------------- | :----- |
| Accuracy          | 0.9442 |
| Balanced Accuracy | 0.9480 |
| Macro F1          | 0.9543 |
| Weighted F1       | 0.9446 |

**Classification Report:**

```
              precision    recall  f1-score   support

 A_PRESERVER       1.00      0.90      0.95      4332
   B_PARTIAL       0.87      1.00      0.93      3999
    C_ERODER       0.99      0.90      0.94      1495
 D_DESTROYER       1.00      0.99      0.99       869

    accuracy                           0.94     10695
   macro avg       0.96      0.95      0.95     10695
weighted avg       0.95      0.94      0.94     10695
```

### XGBoost

| Metric            | Value  |
| :---------------- | :----- |
| Accuracy          | 0.9402 |
| Balanced Accuracy | 0.9404 |
| Macro F1          | 0.9487 |
| Weighted F1       | 0.9406 |

**Classification Report:**

```
              precision    recall  f1-score   support

 A_PRESERVER       1.00      0.90      0.95      4332
   B_PARTIAL       0.87      1.00      0.93      3999
    C_ERODER       0.98      0.89      0.93      1495
 D_DESTROYER       1.00      0.98      0.99       869

    accuracy                           0.94     10695
   macro avg       0.96      0.94      0.95     10695
weighted avg       0.95      0.94      0.94     10695
```

### Random Forest

| Metric            | Value  |
| :---------------- | :----- |
| Accuracy          | 0.9035 |
| Balanced Accuracy | 0.8794 |
| Macro F1          | 0.8992 |
| Weighted F1       | 0.9023 |

**Classification Report:**

```
              precision    recall  f1-score   support

 A_PRESERVER       1.00      0.89      0.94      4332
   B_PARTIAL       0.80      1.00      0.89      3999
    C_ERODER       0.98      0.65      0.78      1495
 D_DESTROYER       0.99      0.98      0.99       869

    accuracy                           0.90     10695
   macro avg       0.94      0.88      0.90     10695
weighted avg       0.92      0.90      0.90     10695
```

---

## 5. Ensemble Model

The ensemble (average of LightGBM and XGBoost probabilities) showed competitive performance:

*   **Accuracy:** 0.9433
*   **Macro F1:** 0.9531

---

## 6. Feature Importance Analysis (LightGBM)

**Top 20 Features by Gain:**

| Feature                       | Importance Gain |
| :---------------------------- | :-------------- |
| PPP_Q_Composite_Score         | 993051.90       |
| Market_Cap_Saturation_Pct     | 93612.07        |
| Composite_Score_5Y            | 65281.58        |
| Vol_Adj_PP_Score_5Y           | 34712.41        |
| Volatility_90D                | 29801.63        |
| Growth_Potential_Multiplier   | 28843.86        |
| Real_Return_5Y                | 22282.84        |
| Calmar_Ratio_5Y               | 21017.84        |
| Real_Return_Milk_1Y           | 15781.45        |
| Recovery_Strength             | 9443.13         |
| Days_Since_ATH                | 8317.12         |
| Return_Consistency            | 6280.05         |
| Real_Return_Eggs_1Y           | 6089.70         |
| Real_Return_3Y                | 5841.39         |
| Sharpe_Ratio_5Y               | 4344.95         |
| Max_Drawdown                  | 3513.01         |
| Distance_From_ATH_Pct         | 3121.04         |
| CPI_Correlation               | 2974.65         |
| Real_PP_Index                 | 2895.18         |
| PP_Multiplier_1Y              | 2882.96         |

---

## 7. Actionable Insights Generation

10,695 actionable investment insights were generated for the test set. A sample insight for Bitcoin is:

*   **Asset:** Bitcoin
*   **Predicted Class:** B_PARTIAL
*   **Confidence:** 100.0%
*   **Volatility:** HIGH
*   **Entry Signal:** CONSIDER
*   **Cycle Position:** VALUE_ZONE

---

## 8. Saved Models & Artifacts

All models and related artifacts have been saved:

*   **LightGBM Model:** `models/pppq/lgbm_model.txt`
*   **XGBoost Model:** `models/pppq/xgb_model.json`
*   **Random Forest Model:** `models/pppq/rf_model.pkl`
*   **Label Encoder:** `models/pppq/label_encoder.pkl`
*   **Feature Columns:** `models/pppq/feature_columns.json`
*   **Feature Importance Report:** `reports/pppq/feature_importance.csv`
*   **Training Summary:** `reports/pppq/training_summary.json`
*   **Investment Insights:** `reports/pppq/investment_insights.csv`

---

## 9. Final Summary

**Model Performance Comparison (Test Set):**

| Model           | Macro F1 | Accuracy |
| :-------------- | :------- | :------- |
| LightGBM        | 0.9543   | 0.9442   |
| XGBoost         | 0.9487   | 0.9402   |
| Random Forest   | 0.8992   | 0.9035   |
| Ensemble        | 0.9531   | 0.9433   |

**🏆 BEST MODEL: LightGBM**

**Output Files Location:**

*   **Models:** `models/pppq/`
*   **Reports:** `reports/pppq/`
*   **Insights:** `reports/pppq/investment_insights.csv`