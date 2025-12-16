"""
Retrain models with updated balanced classification
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import json
from pathlib import Path

print('Loading updated data...')
train_df = pd.read_csv('data/processed/pppq/train/pppq_train.csv')
val_df = pd.read_csv('data/processed/pppq/val/pppq_val.csv')
test_df = pd.read_csv('data/processed/pppq/test/pppq_test.csv')

print(f'Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')
print(f'Train classes: {train_df["PPP_Q_Class"].value_counts().to_dict()}')

# Prepare features
exclude_cols = ['Date', 'Asset', 'PPP_Q_Class', 'Asset_Category', 'PPP_Q_Composite_Score']
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

X_train = train_df[feature_cols].fillna(0)
y_train = train_df['PPP_Q_Class']
X_val = val_df[feature_cols].fillna(0)
y_val = val_df['PPP_Q_Class']
X_test = test_df[feature_cols].fillna(0)
y_test = test_df['PPP_Q_Class']

# Encode labels
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(y_test)

print(f'Features: {len(feature_cols)}')
print(f'Classes: {label_encoder.classes_}')

# Train LightGBM
print('\nTraining LightGBM...')
lgb_train = lgb.Dataset(X_train, label=y_train_enc, feature_name=feature_cols)
lgb_val = lgb.Dataset(X_val, label=y_val_enc, reference=lgb_train)

lgb_params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'val'],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

lgb_pred = lgb_model.predict(X_test)
lgb_pred_class = np.argmax(lgb_pred, axis=1)
lgb_f1 = f1_score(y_test_enc, lgb_pred_class, average='macro')
print(f'LightGBM Macro F1: {lgb_f1:.4f}')

# Train XGBoost
print('\nTraining XGBoost...')
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    objective='multi:softprob',
    num_class=4,
    use_label_encoder=False,
    eval_metric='mlogloss',
    early_stopping_rounds=50,
    verbosity=0
)
xgb_model.fit(X_train, y_train_enc, eval_set=[(X_val, y_val_enc)], verbose=False)
xgb_pred = xgb_model.predict(X_test)
xgb_f1 = f1_score(y_test_enc, xgb_pred, average='macro')
print(f'XGBoost Macro F1: {xgb_f1:.4f}')

# Train Random Forest
print('\nTraining Random Forest...')
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train_enc)
rf_pred = rf_model.predict(X_test)
rf_f1 = f1_score(y_test_enc, rf_pred, average='macro')
print(f'Random Forest Macro F1: {rf_f1:.4f}')

# Best model
best_f1 = max(lgb_f1, xgb_f1, rf_f1)
best_name = 'LightGBM' if lgb_f1 == best_f1 else ('XGBoost' if xgb_f1 == best_f1 else 'RF')
print(f'\nBest Model: {best_name} (F1: {best_f1:.4f})')

# Save models
Path('models/pppq').mkdir(parents=True, exist_ok=True)
lgb_model.save_model('models/pppq/lgbm_model.txt')
xgb_model.save_model('models/pppq/xgb_model.json')
with open('models/pppq/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('models/pppq/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('models/pppq/feature_columns.json', 'w') as f:
    json.dump({'features': feature_cols}, f, indent=2)

print('\nModels saved!')

# Classification report for best model
if best_name == 'LightGBM':
    best_pred = lgb_pred_class
elif best_name == 'XGBoost':
    best_pred = xgb_pred
else:
    best_pred = rf_pred

print(f'\nClassification Report ({best_name}):')
print(classification_report(y_test_enc, best_pred, target_names=label_encoder.classes_))
