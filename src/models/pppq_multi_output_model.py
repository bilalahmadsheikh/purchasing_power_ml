# -*- coding: utf-8 -*-
"""
================================================================================
PPP-Q MULTI-OUTPUT MODEL - PRODUCTION TRAINING SYSTEM
================================================================================
Multi-Asset Investment Strategy Classifier with ML-Predicted Component Scores

OUTPUTS:
1. Classification: A/B/C/D (PPP_Q_Class)
2. Component Scores (8 scores):
   - Real PP Score (with commodity purchasing power)
   - Volatility Score
   - Cycle Score
   - Growth Score
   - Consistency Score
   - Recovery Score
   - Risk-Adjusted Score
   - Commodity Score (eggs/milk purchasing power)

Models: LightGBM (Primary) + XGBoost (Ensemble)
Features: 39 (including eggs/milk purchasing power features)

Author: Bilal Ahmad Sheikh
Institution: GIKI
Date: December 2024
================================================================================
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

class MultiOutputConfig:
    """Configuration for multi-output training"""

    # Paths
    DATA_DIR = 'data/processed/pppq/'
    MODEL_DIR = 'models/pppq/'
    REPORTS_DIR = 'reports/pppq/'

    # Primary Model: LightGBM (Best for imbalanced data)
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 7,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
        'is_unbalance': True
    }

    # Component Score Regression Parameters (LightGBM)
    LGBM_REGRESSION_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 7,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }

    # Secondary Model: XGBoost (Ensemble diversity)
    XGB_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'max_depth': 7,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50
    }

    # XGBoost Regression Parameters
    XGB_REGRESSION_PARAMS = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 7,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50
    }

    # Training parameters
    NUM_BOOST_ROUND = 500
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 50

config = MultiOutputConfig()

# Create directories
for directory in [config.MODEL_DIR, config.REPORTS_DIR, config.REPORTS_DIR + 'visualizations/']:
    os.makedirs(directory, exist_ok=True)

print("="*80)
print("  PPP-Q MULTI-OUTPUT MODEL - PRODUCTION TRAINING")
print("="*80)

# ============================================================================
# 1. LOAD & VALIDATE DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING & VALIDATING DATA")
print("="*80)

train_df = pd.read_csv(config.DATA_DIR + 'train/pppq_train.csv')
val_df = pd.read_csv(config.DATA_DIR + 'val/pppq_val.csv')
test_df = pd.read_csv(config.DATA_DIR + 'test/pppq_test.csv')

print(f"\nData loaded:")
print(f"   Train: {train_df.shape}")
print(f"   Val:   {val_df.shape}")
print(f"   Test:  {test_df.shape}")

# Load feature metadata
with open(config.DATA_DIR + 'pppq_features.json', 'r') as f:
    feature_metadata = json.load(f)

print(f"\nFeatures: {feature_metadata['num_features']}")
print(f"   Classes: {feature_metadata['classes']}")
print(f"   Component Targets: {feature_metadata['num_component_targets']}")

# ============================================================================
# 2. PREPARE FEATURES & TARGETS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: PREPARING FEATURES & TARGETS")
print("="*80)

# Feature columns
feature_cols = feature_metadata['features']
component_targets = feature_metadata['component_targets']

print(f"\nFeature columns: {len(feature_cols)}")
print(f"Component target columns: {len(component_targets)}")

# Separate features and targets
X_train = train_df[feature_cols].fillna(0)
y_train_class = train_df['PPP_Q_Class']
y_train_components = train_df[component_targets].fillna(0)

X_val = val_df[feature_cols].fillna(0)
y_val_class = val_df['PPP_Q_Class']
y_val_components = val_df[component_targets].fillna(0)

X_test = test_df[feature_cols].fillna(0)
y_test_class = test_df['PPP_Q_Class']
y_test_components = test_df[component_targets].fillna(0)

# Encode labels for classification
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_class)
y_val_encoded = label_encoder.transform(y_val_class)
y_test_encoded = label_encoder.transform(y_test_class)

class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}

print(f"\nClass Mapping:")
for idx, class_name in class_mapping.items():
    print(f"   {idx}: {class_name}")

# Class distribution
print(f"\nClass Distribution:")
for split_name, y_encoded in [('TRAIN', y_train_encoded), ('VAL', y_val_encoded), ('TEST', y_test_encoded)]:
    print(f"\n   {split_name}:")
    dist = pd.Series(y_encoded).value_counts().sort_index()
    for idx, count in dist.items():
        print(f"   {class_mapping[idx]}: {count:,} ({count/len(y_encoded)*100:.1f}%)")

# ============================================================================
# 3. TRAIN CLASSIFICATION MODEL - LIGHTGBM
# ============================================================================

print("\n" + "="*80)
print("STEP 3: TRAINING CLASSIFICATION MODEL - LIGHTGBM")
print("="*80)

lgb_train_data = lgb.Dataset(X_train, label=y_train_encoded, feature_name=feature_cols)
lgb_val_data = lgb.Dataset(X_val, label=y_val_encoded, reference=lgb_train_data)

print(f"\nTraining LightGBM classifier...")

evals_result = {}
callbacks = [
    lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
    lgb.log_evaluation(period=config.VERBOSE_EVAL),
    lgb.record_evaluation(evals_result)
]

start_time = datetime.now()

lgbm_classifier = lgb.train(
    config.LGBM_PARAMS,
    lgb_train_data,
    num_boost_round=config.NUM_BOOST_ROUND,
    valid_sets=[lgb_train_data, lgb_val_data],
    valid_names=['train', 'val'],
    callbacks=callbacks
)

lgbm_class_time = (datetime.now() - start_time).total_seconds()

print(f"\nLightGBM classifier trained!")
print(f"   Best iteration: {lgbm_classifier.best_iteration}")
print(f"   Training time: {lgbm_class_time:.2f}s")

# ============================================================================
# 4. TRAIN COMPONENT SCORE MODELS - MULTI-OUTPUT REGRESSION
# ============================================================================

print("\n" + "="*80)
print("STEP 4: TRAINING COMPONENT SCORE MODELS (8 REGRESSORS)")
print("="*80)

# We'll train 8 separate LightGBM regressors (one per component score)
component_models = {}
component_metrics = {}

for i, component_name in enumerate(component_targets):
    print(f"\n[{i+1}/8] Training {component_name}...")

    # Prepare data for this component
    lgb_train_comp = lgb.Dataset(X_train, label=y_train_components.iloc[:, i], feature_name=feature_cols)
    lgb_val_comp = lgb.Dataset(X_val, label=y_val_components.iloc[:, i], reference=lgb_train_comp)

    start_time = datetime.now()

    # Train regressor
    component_model = lgb.train(
        config.LGBM_REGRESSION_PARAMS,
        lgb_train_comp,
        num_boost_round=config.NUM_BOOST_ROUND,
        valid_sets=[lgb_train_comp, lgb_val_comp],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=100)
        ]
    )

    train_time = (datetime.now() - start_time).total_seconds()

    # Evaluate on validation set
    val_pred = component_model.predict(X_val, num_iteration=component_model.best_iteration)
    val_true = y_val_components.iloc[:, i]

    rmse = np.sqrt(mean_squared_error(val_true, val_pred))
    r2 = r2_score(val_true, val_pred)

    component_models[component_name] = component_model
    component_metrics[component_name] = {
        'rmse': rmse,
        'r2': r2,
        'train_time': train_time,
        'best_iteration': component_model.best_iteration
    }

    print(f"   RMSE: {rmse:.3f} | R2: {r2:.3f} | Time: {train_time:.1f}s")

print(f"\nAll 8 component models trained!")

# ============================================================================
# 5. TRAIN ENSEMBLE - XGBOOST
# ============================================================================

print("\n" + "="*80)
print("STEP 5: TRAINING ENSEMBLE MODEL - XGBOOST")
print("="*80)

print(f"\nTraining XGBoost classifier...")

start_time = datetime.now()

xgb_classifier = xgb.XGBClassifier(**config.XGB_PARAMS)
xgb_classifier.fit(
    X_train, y_train_encoded,
    eval_set=[(X_val, y_val_encoded)],
    verbose=False
)

xgb_class_time = (datetime.now() - start_time).total_seconds()

print(f"XGBoost classifier trained!")
print(f"   Best iteration: {xgb_classifier.best_iteration}")
print(f"   Training time: {xgb_class_time:.2f}s")

# ============================================================================
# 6. EVALUATE MODELS
# ============================================================================

print("\n" + "="*80)
print("STEP 6: EVALUATING ALL MODELS")
print("="*80)

# Classification evaluation
def evaluate_classifier(model, X, y_true_encoded, model_name, model_type='lgb'):
    """Evaluate classifier with detailed metrics"""

    # Predict
    if model_type == 'lgb':
        y_pred_proba = model.predict(X, num_iteration=model.best_iteration)
    else:  # xgb
        y_pred_proba = model.predict_proba(X)

    y_pred_encoded = np.argmax(y_pred_proba, axis=1)

    # Metrics
    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    balanced_acc = balanced_accuracy_score(y_true_encoded, y_pred_encoded)
    macro_f1 = f1_score(y_true_encoded, y_pred_encoded, average='macro')

    print(f"\n{model_name.upper()} - TEST SET RESULTS")
    print(f"="*80)
    print(f"   Accuracy:          {accuracy:.4f}")
    print(f"   Balanced Accuracy: {balanced_acc:.4f}")
    print(f"   Macro F1:          {macro_f1:.4f} <- PRIMARY METRIC")

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'probabilities': y_pred_proba
    }

# Evaluate classifiers
lgbm_results = evaluate_classifier(lgbm_classifier, X_test, y_test_encoded, 'LightGBM', 'lgb')
xgb_results = evaluate_classifier(xgb_classifier, X_test, y_test_encoded, 'XGBoost', 'xgb')

# Ensemble
ensemble_proba = (lgbm_results['probabilities'] + xgb_results['probabilities']) / 2
ensemble_pred = np.argmax(ensemble_proba, axis=1)
ensemble_acc = accuracy_score(y_test_encoded, ensemble_pred)
ensemble_macro_f1 = f1_score(y_test_encoded, ensemble_pred, average='macro')

print(f"\nENSEMBLE (LightGBM + XGBoost) RESULTS:")
print(f"   Accuracy:    {ensemble_acc:.4f}")
print(f"   Macro F1:    {ensemble_macro_f1:.4f}")

# Component score evaluation
print(f"\n" + "="*80)
print("COMPONENT SCORE MODEL EVALUATION")
print("="*80)

for component_name, metrics in component_metrics.items():
    print(f"\n{component_name}:")
    print(f"   RMSE: {metrics['rmse']:.3f}")
    print(f"   R2:   {metrics['r2']:.3f}")

# ============================================================================
# 7. SAVE MODELS & ARTIFACTS
# ============================================================================

print("\n" + "="*80)
print("STEP 7: SAVING MODELS & ARTIFACTS")
print("="*80)

# Save classification models
lgbm_classifier.save_model(config.MODEL_DIR + 'lgbm_classifier.txt')
xgb_classifier.save_model(config.MODEL_DIR + 'xgb_classifier.json')

# Save component score models
for component_name, model in component_models.items():
    model_filename = f"lgbm_{component_name.lower()}_regressor.txt"
    model.save_model(config.MODEL_DIR + model_filename)

# Save label encoder
with open(config.MODEL_DIR + 'label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save feature columns
with open(config.MODEL_DIR + 'feature_columns.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

# Save component target columns
with open(config.MODEL_DIR + 'component_targets.json', 'w') as f:
    json.dump(component_targets, f, indent=2)

# Save comprehensive results
results_summary = {
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'classification_models': {
        'lightgbm': {
            'macro_f1': float(lgbm_results['macro_f1']),
            'accuracy': float(lgbm_results['accuracy']),
            'training_time_seconds': lgbm_class_time
        },
        'xgboost': {
            'macro_f1': float(xgb_results['macro_f1']),
            'accuracy': float(xgb_results['accuracy']),
            'training_time_seconds': xgb_class_time
        },
        'ensemble': {
            'macro_f1': float(ensemble_macro_f1),
            'accuracy': float(ensemble_acc)
        }
    },
    'component_score_models': {
        comp_name: {
            'rmse': float(metrics['rmse']),
            'r2': float(metrics['r2']),
            'training_time_seconds': float(metrics['train_time'])
        }
        for comp_name, metrics in component_metrics.items()
    },
    'num_features': len(feature_cols),
    'num_component_targets': len(component_targets),
    'classes': label_encoder.classes_.tolist()
}

with open(config.REPORTS_DIR + 'multi_output_training_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nAll models and artifacts saved!")
print(f"   Classification models: {config.MODEL_DIR}lgbm_classifier.txt, xgb_classifier.json")
print(f"   Component models: {len(component_models)} regressors saved")

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

print(f"\nMODEL PERFORMANCE (Test Set):")
print(f"   {'Model':<20} {'Macro F1':<12} {'Accuracy':<12}")
print(f"   {'-'*44}")
print(f"   {'LightGBM':<20} {lgbm_results['macro_f1']:<12.4f} {lgbm_results['accuracy']:<12.4f}")
print(f"   {'XGBoost':<20} {xgb_results['macro_f1']:<12.4f} {xgb_results['accuracy']:<12.4f}")
print(f"   {'Ensemble':<20} {ensemble_macro_f1:<12.4f} {ensemble_acc:<12.4f}")

print(f"\nCOMPONENT SCORE MODELS:")
avg_r2 = np.mean([m['r2'] for m in component_metrics.values()])
avg_rmse = np.mean([m['rmse'] for m in component_metrics.values()])
print(f"   Average R2:   {avg_r2:.3f}")
print(f"   Average RMSE: {avg_rmse:.3f}")

print(f"\nOUTPUT FILES:")
print(f"   Models: {config.MODEL_DIR}")
print(f"   Reports: {config.REPORTS_DIR}")

print("\n" + "="*80)
