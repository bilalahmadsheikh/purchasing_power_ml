# 🧪 Testing Documentation

## Overview (v1.2.0)

This document describes all test categories, what's being tested, and how to run them.

The testing suite covers:
1. **API Tests** - REST endpoints and response validation
2. **ML Tests** - Model validation, drift detection, data validation
3. **Data Tests** - Data quality, schema validation
4. **Integration Tests** - End-to-end workflows
5. **Ensemble Tests** - LightGBM + XGBoost ensemble predictions

---

## v1.2.0 Test Updates

### New Test Categories

#### Ensemble Model Tests
```python
def test_ensemble_prediction():
    """Test ensemble (LightGBM + XGBoost) predictions"""
    # Checks: model_type="ensemble" works correctly
    # Ensures: Both models loaded and averaged

def test_model_type_lgbm():
    """Test LightGBM-only predictions"""
    # Checks: model_type="lgbm" returns valid prediction
    # Ensures: LightGBM standalone works

def test_model_type_xgb():
    """Test XGBoost-only predictions"""
    # Checks: model_type="xgb" returns valid prediction
    # Ensures: XGBoost standalone works

def test_dynamic_weights():
    """Test dynamic weight calculation"""
    # Checks: Weights change based on horizon_years
    # Ensures: <2Y, 2-5Y, 5Y+ have different weights

def test_threshold_classification():
    """Test threshold-based classification"""
    # Checks: Score >= 65 → A_PRESERVER, etc.
    # Ensures: Deterministic classification
```

### Test Command
```bash
# Run all tests
python -m pytest tests/ -v

# Test ensemble specifically
python -m pytest tests/ -v -k "ensemble or model_type"
```

---

## Test Structure

```
tests/
├── test_api.py                 # API endpoint tests
├── test_ml_validation.py       # Model validation tests
├── test_new_endpoints.py       # New prediction endpoints
├── test_data_validation.py     # Data quality tests
└── pytest.ini                  # Pytest configuration

src/ml_testing/
├── data_validation.py          # Data quality checks
├── drift_detection.py          # Feature/target drift detection
└── model_validation.py         # Model performance checks
```

---

## 1️⃣ API Tests (`test_api.py`)

### What's Tested

Tests for REST API endpoints in `src/api/main.py`

#### Test Categories

##### **Health Check Tests**
```python
def test_health():
    """Verify API is healthy and responsive"""
    # Checks: HTTP 200, response format, status = "healthy"
    # Ensures: API server is running and accessible

def test_root():
    """Verify root endpoint"""
    # Checks: HTTP 200, returns welcome message
    # Ensures: Basic connectivity

def test_version():
    """Check API version"""
    # Checks: Version string format (e.g., "1.0.0")
    # Ensures: Version tracking works
```

##### **Model Info Tests**
```python
def test_get_model_info():
    """Retrieve model metadata"""
    # Checks: HTTP 200, contains version, metrics, deployment_date
    # Ensures: Model info endpoint works

def test_get_feature_columns():
    """Get list of features used by model"""
    # Checks: HTTP 200, array of feature names
    # Ensures: Feature list is accurate

def test_get_asset_classes():
    """Retrieve supported asset classes"""
    # Checks: HTTP 200, contains all 4 PPP_Q classes
    # Ensures: Classification scheme is documented
```

##### **Prediction Tests (Single Request)**
```python
def test_predict_bitcoin():
    """Predict PPP_Q class for Bitcoin"""
    # Input: Bitcoin features for a specific date
    # Checks: HTTP 200, class in [A, B, C, D], confidence scores
    # Ensures: Single prediction works correctly

def test_predict_gold():
    """Predict PPP_Q class for Gold"""
    # Similar to Bitcoin but for Gold

def test_predict_sp500():
    """Predict PPP_Q class for S&P 500"""
    # Similar to Bitcoin but for S&P 500
```

##### **Batch Prediction Tests**
```python
def test_batch_predict():
    """Predict for multiple assets"""
    # Input: 5 different assets with their features
    # Checks: HTTP 200, returns predictions for all 5
    # Ensures: Batch processing works

def test_batch_predict_with_confidence():
    """Batch predictions with confidence scores"""
    # Checks: confidence scores sum to 1.0 for each prediction
    # Ensures: Probability calibration
```

##### **Error Handling Tests**
```python
def test_predict_missing_features():
    """Handle missing required features"""
    # Input: Request with missing features
    # Checks: HTTP 422, error message about missing fields
    # Ensures: Validation catches bad requests

def test_predict_invalid_asset():
    """Handle invalid asset name"""
    # Input: Asset = "UnknownCoin"
    # Checks: HTTP 422, error message
    # Ensures: Asset validation works

def test_predict_out_of_range_features():
    """Handle out-of-range feature values"""
    # Input: Volatility = 500 (unrealistic)
    # Checks: HTTP 422 or accepts with warning
    # Ensures: Bounds checking

def test_rate_limiting():
    """Verify rate limiting works"""
    # Input: 100 requests in quick succession
    # Checks: Some requests return 429 (Too Many Requests)
    # Ensures: API is protected from abuse
```

##### **Response Format Tests**
```python
def test_prediction_response_schema():
    """Verify response matches OpenAPI schema"""
    # Checks: Response has correct fields and types
    # Fields: asset, prediction, confidence, probability_distribution
    # Ensures: Client contract is maintained

def test_batch_response_format():
    """Verify batch response format"""
    # Checks: Array of predictions, metadata
    # Ensures: Batch response structure is correct
```

### Running API Tests

```bash
# Run all API tests
pytest tests/test_api.py -v

# Run specific test
pytest tests/test_api.py::test_predict_bitcoin -v

# Run with coverage
pytest tests/test_api.py --cov=src.api --cov-report=html

# Run and generate report
pytest tests/test_api.py -v --tb=short > api_test_results.txt
```

### Expected Results
- ✅ All tests pass (HTTP 200 for success cases)
- ✅ Error cases handled gracefully (HTTP 422 for validation)
- ✅ Response times < 500ms per prediction
- ✅ Batch predictions faster than individual

---

## 2️⃣ ML Validation Tests (`test_ml_validation.py`)

### What's Tested

Model performance, validation, and machine learning specific checks.

#### Test Categories

##### **Model Loading Tests**
```python
def test_load_lightgbm_model():
    """Load LightGBM model from disk"""
    # Checks: Model loads without errors
    # Ensures: Model file integrity

def test_load_xgboost_model():
    """Load XGBoost model from disk"""
    # Checks: Model loads without errors
    # Ensures: Model file integrity

def test_load_ensemble_model():
    """Load ensemble from component models"""
    # Checks: All 3 component models load
    # Ensures: Ensemble can be constructed
```

##### **Model Prediction Tests**
```python
def test_lightgbm_predictions():
    """Generate predictions with LightGBM"""
    # Checks: Produces output for all test samples
    # Ensures: Model inference works

def test_xgboost_predictions():
    """Generate predictions with XGBoost"""
    # Checks: Produces output for all test samples
    # Ensures: Model inference works

def test_ensemble_predictions():
    """Generate predictions with ensemble"""
    # Checks: Weighted average of component models
    # Ensures: Ensemble combines models correctly

def test_prediction_consistency():
    """Same input produces same output"""
    # Input: Run prediction 3 times with same data
    # Checks: All 3 outputs are identical
    # Ensures: Model is deterministic
```

##### **Performance Metrics Tests**
```python
def test_model_accuracy():
    """Check overall accuracy on test set"""
    # Checks: accuracy >= 0.65 (65%)
    # Ensures: Minimum performance threshold

def test_macro_f1_score():
    """Check macro-averaged F1 on test set"""
    # Checks: macro_f1 >= 0.70
    # Ensures: Good performance across all classes

def test_balanced_accuracy():
    """Check balanced accuracy (per-class recall)"""
    # Checks: balanced_accuracy >= 0.60
    # Ensures: No class is completely missed

def test_per_class_performance():
    """Check F1 for each class separately"""
    # Checks:
    #   - A_PRESERVER: F1 >= 0.70
    #   - B_PARTIAL: F1 >= 0.65
    #   - C_ERODER: F1 >= 0.60
    #   - D_DESTROYER: F1 >= 0.55
    # Ensures: Each class has adequate performance
```

##### **Calibration Tests**
```python
def test_probability_distribution():
    """Check that probabilities are calibrated"""
    # Checks: Sum of class probabilities = 1.0 (±0.001)
    # Ensures: Confidence scores are valid

def test_confidence_distribution():
    """Check confidence score distribution"""
    # Checks: No impossible confidence values (>1 or <0)
    # Ensures: Probabilities are bounded [0,1]

def test_class_probability_ordering():
    """Check predicted class has highest probability"""
    # Checks: argmax(probabilities) == predicted_class
    # Ensures: Predictions are consistent
```

##### **Feature Importance Tests**
```python
def test_feature_importance_shape():
    """Check feature importance output shape"""
    # Checks: One importance score per feature
    # Ensures: All features have importance

def test_feature_importance_bounds():
    """Check importance scores are valid"""
    # Checks: All scores >= 0, sum approximately = 1
    # Ensures: Importance is normalized

def test_top_features():
    """Check that top features make sense"""
    # Checks: Top 5 features contain expected vars
    # Expected: volatility, purchasing power, macroeconomic vars
    # Ensures: Model focuses on interpretable features
```

##### **Robustness Tests**
```python
def test_predictions_with_noise():
    """Check model robustness to small input perturbations"""
    # Process:
    #   1. Get base prediction
    #   2. Add small Gaussian noise to features
    #   3. Get prediction with noise
    #   4. Check predictions are similar
    # Threshold: Class should remain same in 90%+ of noise samples
    # Ensures: Model isn't overly sensitive

def test_extreme_values():
    """Check model handles extreme but valid values"""
    # Input: Very high/low volatility, returns, etc.
    # Checks: Model produces reasonable predictions
    # Ensures: No crashes on edge cases

def test_missing_data_handling():
    """Check how model handles NaN values"""
    # Input: Features with NaN (imputed with mean)
    # Checks: Model produces predictions
    # Ensures: Graceful degradation
```

### Running ML Tests

```bash
# Run all ML validation tests
pytest tests/test_ml_validation.py -v

# Run specific test category
pytest tests/test_ml_validation.py -k "performance" -v

# Run with detailed output
pytest tests/test_ml_validation.py -v -s

# Generate report
pytest tests/test_ml_validation.py --html=ml_report.html
```

### Expected Results
- ✅ Accuracy >= 65%
- ✅ Macro F1 >= 0.70
- ✅ All models load correctly
- ✅ Predictions are deterministic
- ✅ Probabilities sum to 1.0
- ✅ Model is robust to noise

---

## 3️⃣ Data Validation Tests (`test_data_validation.py`)

### What's Tested

Data quality, schema validation, and data consistency.

#### Test Categories

##### **Data Schema Tests**
```python
def test_consolidated_dataset_schema():
    """Check final_consolidated_dataset.csv has correct columns"""
    # Checks:
    #   - Has 'Date' column
    #   - Has price/return columns for all assets
    #   - Has all economic indicator columns
    # Ensures: Expected data structure

def test_processed_data_schema():
    """Check processed PPPQ data schema"""
    # Checks:
    #   - Has 'Date', 'Asset', 'Asset_Category'
    #   - Has all feature columns
    #   - Has 'PPP_Q_Class' column
    # Ensures: Features are complete

def test_train_val_test_schema():
    """Check train/val/test splits have same schema"""
    # Checks: Columns are identical across all 3 sets
    # Ensures: Consistent data format
```

##### **Data Type Tests**
```python
def test_date_column_is_datetime():
    """Verify Date column is datetime format"""
    # Checks: Date column parses correctly
    # Ensures: Time-based operations work

def test_numeric_columns_are_numeric():
    """Verify numeric columns have correct dtype"""
    # Checks: Returns, prices, indicators are float64
    # Ensures: Arithmetic operations work

def test_categorical_columns_are_categorical():
    """Verify categorical columns are string/category"""
    # Checks: Asset, Asset_Category, PPP_Q_Class are strings
    # Ensures: Classification works correctly
```

##### **Missing Value Tests**
```python
def test_no_missing_dates():
    """Check for gaps in date range"""
    # Checks: Date range is continuous (or acceptable gaps)
    # Tolerance: Max gap <= 5 business days (weekends)
    # Ensures: No missing time periods

def test_minimal_missing_features():
    """Check missing values in features"""
    # Checks: < 5% missing values overall
    # Checks: Key features < 1% missing
    # Ensures: Data completeness

def test_missing_values_imputed():
    """Verify missing values are handled"""
    # Checks: No NaN values in final datasets
    # Method: Mean/forward-fill imputation
    # Ensures: Ready for training
```

##### **Value Range Tests**
```python
def test_prices_are_positive():
    """Check all prices > 0"""
    # Checks: No negative or zero prices
    # Ensures: Economic validity

def test_returns_are_reasonable():
    """Check returns within reasonable bounds"""
    # Checks: Returns between -100% and +1000%
    # Ensures: No data entry errors

def test_volatility_is_non_negative():
    """Check volatility >= 0"""
    # Checks: No negative volatility
    # Ensures: Statistical validity

def test_sharpe_ratio_bounds():
    """Check Sharpe ratios are reasonable"""
    # Checks: Between -5 and +5
    # Ensures: No calculation errors
```

##### **Duplicate Tests**
```python
def test_no_duplicate_dates_per_asset():
    """Check for duplicate rows"""
    # Checks: Each (Date, Asset) pair appears once
    # Ensures: No accidental duplicates

def test_no_duplicate_rows_overall():
    """Check for completely identical rows"""
    # Checks: No exact duplicates in dataset
    # Ensures: Data integrity
```

##### **Distribution Tests**
```python
def test_class_distribution_balanced():
    """Check class balance in train/val/test"""
    # Checks:
    #   - No class < 10% of total
    #   - No class > 40% of total
    # Ensures: Balanced training

def test_asset_distribution():
    """Check asset representation"""
    # Checks: All 12+ assets present
    # Ensures: Complete asset coverage

def test_temporal_distribution():
    """Check data is distributed across time"""
    # Checks: Samples from all years in range
    # Ensures: No time-clustered data
```

##### **Consistency Tests**
```python
def test_train_val_test_no_overlap():
    """Check train/val/test don't overlap in time"""
    # Checks: test_end < val_start < train_end
    # Ensures: No data leakage

def test_feature_consistency():
    """Check features have same scale across splits"""
    # Checks: Mean, std dev similar in train/val/test
    # Ensures: Consistent distributions

def test_class_ratio_consistency():
    """Check class distributions are similar"""
    # Checks: Class ratios similar in train/val/test
    # Ensures: Representative splits
```

### Running Data Tests

```bash
# Run all data validation tests
pytest tests/test_data_validation.py -v

# Run specific category
pytest tests/test_data_validation.py -k "schema" -v

# Check specific dataset
pytest tests/test_data_validation.py::test_consolidated_dataset_schema -v

# Generate data quality report
pytest tests/test_data_validation.py --html=data_quality_report.html
```

### Expected Results
- ✅ All required columns present
- ✅ Correct data types
- ✅ < 5% missing values
- ✅ No data entry errors (unrealistic values)
- ✅ No duplicates
- ✅ Balanced class distribution
- ✅ No temporal leakage

---

## 4️⃣ Integration Tests (`test_new_endpoints.py`)

### What's Tested

End-to-end workflows and complex scenarios.

#### Test Categories

##### **End-to-End Prediction Tests**
```python
def test_full_prediction_pipeline():
    """Test complete prediction workflow"""
    # Steps:
    #   1. Fetch latest data
    #   2. Preprocess data
    #   3. Generate prediction
    #   4. Return result
    # Checks: Success at each step
    # Ensures: Complete pipeline works

def test_batch_prediction_pipeline():
    """Test batch prediction workflow"""
    # Steps:
    #   1. Load multiple assets
    #   2. Generate batch predictions
    #   3. Aggregate results
    # Checks: All predictions generated
    # Ensures: Batch processing works
```

##### **Model Comparison Tests**
```python
def test_all_models_agree():
    """Check model consistency"""
    # Process:
    #   1. Run LightGBM prediction
    #   2. Run XGBoost prediction
    #   3. Run Random Forest prediction
    #   4. Check all predict same class
    # Tolerance: 90%+ agreement
    # Ensures: Model consensus

def test_ensemble_is_better():
    """Check ensemble improves over base models"""
    # Metric: F1 score
    # Checks: Ensemble F1 >= max(base model F1s)
    # Ensures: Ensemble provides value
```

##### **API-ML Integration Tests**
```python
def test_api_returns_same_as_model():
    """API predictions match direct model calls"""
    # Process:
    #   1. Call API endpoint
    #   2. Call model directly
    #   3. Compare results
    # Checks: Identical predictions
    # Ensures: No discrepancies in API layer

def test_api_model_version_matches():
    """API serves correct model version"""
    # Checks: Model version in API == on disk
    # Ensures: Correct model is deployed
```

##### **Error Recovery Tests**
```python
def test_graceful_degradation():
    """Handle data issues gracefully"""
    # Scenario: Missing features
    # Check: API returns 422 with clear message
    # Ensures: Helpful error messages

def test_timeout_handling():
    """Handle slow requests"""
    # Scenario: Very large batch (1000 samples)
    # Check: Request times out gracefully
    # Ensures: No hung processes

def test_concurrent_requests():
    """Handle multiple simultaneous requests"""
    # Process: 10 concurrent predictions
    # Check: All complete without errors
    # Ensures: Thread safety
```

##### **Model Update Tests**
```python
def test_model_hot_reload():
    """Load new model without API restart"""
    # Process:
    #   1. Generate new model
    #   2. Reload in API
    #   3. Generate prediction
    # Check: Prediction uses new model
    # Ensures: Zero-downtime updates

def test_rollback_to_previous_model():
    """Ability to rollback model version"""
    # Process:
    #   1. Save previous model as fallback
    #   2. Deploy new model
    #   3. Rollback to previous
    # Check: Predictions revert to old model
    # Ensures: Safe model deployment
```

### Running Integration Tests

```bash
# Run all integration tests
pytest tests/test_new_endpoints.py -v

# Run specific scenario
pytest tests/test_new_endpoints.py -k "pipeline" -v

# Run with API server (requires separate terminal)
pytest tests/test_new_endpoints.py --live-server -v

# Generate integration report
pytest tests/test_new_endpoints.py --html=integration_report.html
```

### Expected Results
- ✅ Complete pipelines execute without errors
- ✅ API-ML results are consistent
- ✅ Models generally agree (>90%)
- ✅ Ensemble improves performance
- ✅ Errors handled gracefully
- ✅ Concurrent requests work

---

## 5️⃣ Additional ML Testing Modules

### Data Validation (`src/ml_testing/data_validation.py`)

```python
# Checks implemented:
check_missing_values()
  - Reports % missing per column
  - Flags high-missing columns

check_data_types()
  - Verifies dtype correctness
  - Reports type mismatches

check_value_ranges()
  - Detects outliers (>3 sigma)
  - Flags unrealistic values

check_duplicates()
  - Finds duplicate rows
  - Suggests removal

validate_schema()
  - Ensures required columns
  - Validates column order
```

### Drift Detection (`src/ml_testing/drift_detection.py`)

```python
# Detects data/model drift:
detect_feature_drift()
  - Compares feature distributions
  - Kolmogorov-Smirnov test
  - Alerts if p < 0.05

detect_target_drift()
  - Checks class distribution changes
  - Chi-square test
  - Alerts if significant shift

detect_prediction_drift()
  - Compares prediction distributions
  - Identifies model behavior changes
  
detect_covariate_shift()
  - Detects input distribution change
  - May indicate new market regime
```

### Model Validation (`src/ml_testing/model_validation.py`)

```python
# Validates model quality:
validate_performance()
  - Checks against thresholds
  - F1 >= 0.70
  - Accuracy >= 0.65

validate_reproducibility()
  - Same input → same output
  - Model determinism check

validate_fairness()
  - Performance across asset classes
  - Per-class F1 scores

validate_predictions()
  - Range validation
  - Probability calibration
  - Confidence distributions
```

---

## Running All Tests

### Run Everything
```bash
# Run all tests with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov=tests --cov-report=html

# Run with specific markers
pytest -m "api or ml" -v
```

### Generate Test Reports
```bash
# HTML report
pytest --html=report.html --self-contained-html

# XML report (for CI/CD)
pytest --junit-xml=test_results.xml

# Coverage report
pytest --cov=src --cov-report=term-missing --cov-report=html
```

### Test Configuration (pytest.ini)
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    api: API endpoint tests
    ml: Machine learning tests
    data: Data validation tests
    slow: Slow tests
    requires_data: Tests requiring full dataset
```

---

## CI/CD Integration

### GitHub Actions Workflow
```yaml
# Runs on: Every push to develop/main
# Tests:
#   1. All unit tests
#   2. Data validation
#   3. ML validation
#   4. API integration tests
# Report: Coverage badge, test artifacts
```

### Test Gating
```
Pull Request → Run Tests → Report Results → Merge?
                  ↓
         Coverage >= 80%? ✅
         All tests pass? ✅
         No new issues? ✅
                  ↓
              Merge Allowed
```

---

## Test Coverage

### Current Coverage (Target: >80%)
```
src/api/               85% (main.py: 88%, schemas.py: 82%)
src/data/              75% (data_collection.py: 70%, preprocessing: 80%)
src/models/            80% (pppq_model.py: 82%)
src/pipelines/         70% (prefect_flows.py: 65%, config: 90%)
src/ml_testing/        85% (all modules: 85%+)
```

### Coverage Report
```bash
# Generate coverage report
pytest --cov=src --cov-report=html
# Open: htmlcov/index.html
```

---

## Best Practices

### Writing Tests
1. **Clear naming:** `test_module_scenario_expected`
2. **Arrange-Act-Assert:** Setup → Execute → Verify
3. **Single responsibility:** One thing per test
4. **Isolated:** Independent of other tests
5. **Deterministic:** Same result every time

### Example Test
```python
def test_predict_bitcoin():
    """Predict PPP_Q class for Bitcoin given features"""
    # Arrange
    features = {
        'pp_multiplier_5y': 2.5,
        'volatility_90d': 45.0,
        'sharpe_ratio_5y': 0.8,
    }
    
    # Act
    response = client.post("/api/v1/predict", json=features)
    
    # Assert
    assert response.status_code == 200
    assert response.json()['class'] in ['A', 'B', 'C', 'D']
    assert 0 <= response.json()['confidence'] <= 1
```

---

## Debugging Failing Tests

```bash
# Run with print statements
pytest -v -s

# Run specific failing test
pytest tests/test_api.py::test_predict_bitcoin -v

# Run with pdb debugger
pytest --pdb tests/test_api.py::test_predict_bitcoin

# Show local variables
pytest -v --tb=short

# Keep temp files
pytest --basetemp=/tmp/pytest
```

---

## Performance Benchmarks

### Expected Test Runtimes
| Category | Count | Time |
|----------|-------|------|
| API Tests | 15 | ~5 sec |
| ML Tests | 20 | ~30 sec |
| Data Tests | 25 | ~10 sec |
| Integration | 10 | ~20 sec |
| **Total** | **70** | **~65 sec** |

---

**Last Updated:** December 2024  
**Test Suite Version:** 1.0  
**Coverage Target:** 80%+  
**Maintenance:** Run before each commit
