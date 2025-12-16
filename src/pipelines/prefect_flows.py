"""
================================================================================
PREFECT ML PIPELINE ORCHESTRATION
================================================================================
Complete Prefect pipeline for PPP-Q ML model

Tasks:
1. Data Ingestion (fetch new data)
2. Feature Engineering (preprocess)
3. Model Training (LightGBM, XGBoost, RF)
4. Evaluation (metrics & comparison)
5. Model Versioning (MLflow)
6. Notifications (Email)

Schedule: Every 15 days

Author: Bilal Ahmad Sheikh
Date: December 2024
================================================================================
"""

import logging
import sys
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Prefect imports
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash

# Local imports
from .pipeline_config import PipelineConfig
from .notifications import notifier
from .model_registry import registry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PipelineConfig.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

config = PipelineConfig


# ============================================================================
# TASK 1: DATA INGESTION
# ============================================================================

@task(
    name="fetch_new_data",
    retries=config.RETRY_ATTEMPTS,
    retry_delay_seconds=config.RETRY_DELAY_SECONDS,
    timeout_seconds=config.TIMEOUT_SECONDS
)
def fetch_new_data() -> Tuple[pd.DataFrame, int]:
    """
    Fetch new data from all sources
    Only fetches data newer than what exists
    
    Returns:
        Tuple[DataFrame, int]: (complete_data, new_rows_count)
    """
    logger.info("="*80)
    logger.info("TASK 1: DATA INGESTION")
    logger.info("="*80)
    
    try:
        # Import data collection functions
        sys.path.insert(0, str(config.PROJECT_ROOT))
        from src.data.data_collection import (
            fetch_economic_data,
            fetch_asset_and_vix_prices,
            fetch_crypto_data_yfinance,
            fetch_crypto_supply_yfinance,
            fetch_real_baselines,
            fetch_global_market_data,
            merge_all_raw_data,
            calculate_all_returns_volatility_technicals,
            engineer_core_features,
            add_purchasing_power_multipliers_and_baselines,
            add_economic_rules,
            add_market_cap_saturation_and_risk,
            create_labels,
            round_all_numerical_columns,
            get_all_asset_configs
        )
        
        # Check existing data
        existing_rows = 0
        last_date = None
        if config.RAW_DATA_PATH.exists():
            existing_df = pd.read_csv(config.RAW_DATA_PATH)
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
            existing_rows = len(existing_df)
            last_date = existing_df['Date'].max()
            logger.info(f"📊 Existing data: {existing_rows} rows (last: {last_date.date()})")
        
        # Fetch fresh data
        logger.info("🔄 Fetching economic data...")
        df_economic = fetch_economic_data()
        
        logger.info("🔄 Fetching asset prices...")
        df_assets_vix = fetch_asset_and_vix_prices()
        
        logger.info("🔄 Fetching crypto data...")
        df_crypto_prices = fetch_crypto_data_yfinance()
        df_crypto_supply = fetch_crypto_supply_yfinance()
        
        logger.info("🔄 Fetching commodity baselines...")
        df_commodities = fetch_real_baselines()
        
        logger.info("🔄 Fetching global market data...")
        df_global_m2, df_global_gdp = fetch_global_market_data()
        
        # Merge all data
        logger.info("🔗 Merging all data sources...")
        df_merged = merge_all_raw_data(
            df_economic, df_assets_vix, df_crypto_prices,
            df_commodities, df_global_m2, df_global_gdp
        )
        
        # Get crypto supplies for asset configs
        crypto_supplies = {}
        if df_crypto_supply is not None and not df_crypto_supply.empty:
            for idx, row in df_crypto_supply.iterrows():
                crypto_supplies[row['Asset']] = {
                    'circulating': row.get('Circulating_Supply', 1e7),
                    'max': row.get('Max_Supply', 2.1e7)
                }
        
        # Feature engineering
        logger.info("⚙️ Engineering features...")
        asset_configs = get_all_asset_configs(df_merged, crypto_supplies)
        
        df_featured = calculate_all_returns_volatility_technicals(df_merged, asset_configs)
        df_featured = engineer_core_features(df_featured)
        df_featured = add_purchasing_power_multipliers_and_baselines(df_featured, asset_configs)
        df_featured = add_economic_rules(df_featured)
        df_featured = add_market_cap_saturation_and_risk(
            df_featured, None, None, df_crypto_supply
        )
        df_featured = create_labels(df_featured, asset_configs)
        df_featured = round_all_numerical_columns(df_featured, 4)
        
        # Save consolidated dataset
        df_featured.to_csv(config.RAW_DATA_PATH, index=False)
        
        new_rows = len(df_featured) - existing_rows
        logger.info(f"✅ Data ingestion complete: {len(df_featured)} total rows ({new_rows} new)")
        
        return df_featured, new_rows
    
    except Exception as e:
        logger.error(f"❌ Data ingestion failed: {str(e)}")
        raise


# ============================================================================
# TASK 2: PREPROCESSING & FEATURE ENGINEERING
# ============================================================================

@task(
    name="preprocess_data",
    retries=config.RETRY_ATTEMPTS,
    retry_delay_seconds=config.RETRY_DELAY_SECONDS,
    timeout_seconds=config.TIMEOUT_SECONDS
)
def preprocess_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data and create train/val/test splits
    
    Returns:
        Tuple[train_df, val_df, test_df]
    """
    logger.info("="*80)
    logger.info("TASK 2: PREPROCESSING & FEATURE ENGINEERING")
    logger.info("="*80)
    
    try:
        # Import preprocessing logic
        sys.path.insert(0, str(config.PROJECT_ROOT))
        
        df = df_raw.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # ================================================================
        # PPPQ PREPROCESSING (from preprocessing_pppq.py logic)
        # ================================================================
        
        def get_asset_category(asset):
            if asset in config.CRYPTO_ASSETS:
                return 'crypto'
            elif asset in config.PRECIOUS_METALS:
                return 'metal'
            elif asset in config.COMMODITIES:
                return 'commodity'
            elif asset in config.EQUITY_INDICES:
                return 'index'
            elif asset in config.TECH_STOCKS:
                return 'stock'
            else:
                return 'etf'
        
        def round_to_3dp(value):
            if pd.isna(value):
                return 0.0
            return round(float(value), 3)
        
        all_asset_data = []
        
        for asset in config.CORE_ASSETS:
            logger.info(f"   Processing: {asset}")
            
            asset_category = get_asset_category(asset)
            asset_df = df[['Date']].copy()
            asset_df['Asset'] = asset
            asset_df['Asset_Category'] = asset_category
            
            # Extract features for this asset
            for horizon in ['3Y', '5Y', '10Y']:
                col = f'{asset}_Real_Return_{horizon}'
                asset_df[f'Real_Return_{horizon}'] = df[col].apply(round_to_3dp) if col in df.columns else 0.0
            
            for horizon in ['1Y', '5Y', '10Y']:
                col = f'{asset}_PP_Multiplier_{horizon}'
                asset_df[f'PP_Multiplier_{horizon}'] = df[col].apply(round_to_3dp) if col in df.columns else 1.0
            
            # Volatility & Risk
            vol_col = f'{asset}_Volatility_90D'
            asset_df['Volatility_90D'] = df[vol_col].apply(round_to_3dp) if vol_col in df.columns else 0.0
            
            sharpe_col = f'{asset}_Sharpe_Ratio_5Y'
            asset_df['Sharpe_Ratio_5Y'] = df[sharpe_col].apply(round_to_3dp) if sharpe_col in df.columns else 0.0
            
            dd_col = f'{asset}_Max_Drawdown'
            asset_df['Max_Drawdown'] = df[dd_col].abs().apply(round_to_3dp) if dd_col in df.columns else 0.0
            
            # Market cycle
            price_col = f'{asset}_Price' if f'{asset}_Price' in df.columns else asset
            if price_col in df.columns:
                prices = df[price_col].values
                ath = np.maximum.accumulate(prices)
                asset_df['Distance_From_ATH_Pct'] = np.where(ath > 0, ((prices - ath) / ath) * 100, 0).round(3)
            else:
                asset_df['Distance_From_ATH_Pct'] = 0.0
            
            ma_200_col = f'{asset}_MA_200D'
            if ma_200_col in df.columns and price_col in df.columns:
                ma_200 = df[ma_200_col].values
                prices = df[price_col].values
                asset_df['Distance_From_MA_200D_Pct'] = np.where(
                    ma_200 > 0, ((prices - ma_200) / ma_200) * 100, 0
                ).round(3)
            else:
                asset_df['Distance_From_MA_200D_Pct'] = 0.0
            
            # Saturation
            sat_col = f'{asset}_Market_Cap_Saturation_Pct'
            asset_df['Market_Cap_Saturation_Pct'] = df[sat_col].apply(round_to_3dp) if sat_col in df.columns else 50.0
            
            # Composite score
            comp_col = f'{asset}_Composite_Score_5Y'
            asset_df['Composite_Score_5Y'] = df[comp_col].apply(round_to_3dp) if comp_col in df.columns else 50.0
            
            # Additional features
            asset_df['PP_Stability_Index'] = 0.5
            asset_df['Return_Consistency'] = 0.5
            asset_df['Recovery_Strength'] = 0.5
            asset_df['Days_Since_ATH'] = 0
            
            all_asset_data.append(asset_df)
        
        # Combine all assets
        pppq_df = pd.concat(all_asset_data, ignore_index=True)
        
        # Calculate composite scores and classes
        def calculate_composite_score(row):
            pp_mult = row.get('PP_Multiplier_5Y', 1.0)
            volatility = row.get('Volatility_90D', 0)
            saturation = row.get('Market_Cap_Saturation_Pct', 50)
            sharpe = row.get('Sharpe_Ratio_5Y', 0)
            distance_ath = row.get('Distance_From_ATH_Pct', 0)
            
            # PP Score (25%)
            if pp_mult < 0.85:
                pp_score = 20
            elif pp_mult < 1.0:
                pp_score = 40
            elif pp_mult < 1.3:
                pp_score = 60
            elif pp_mult < 2.0:
                pp_score = 80
            else:
                pp_score = 100
            
            # Volatility Score (20%)
            if volatility < 15:
                vol_score = 100
            elif volatility < 25:
                vol_score = 75
            elif volatility < 40:
                vol_score = 50
            else:
                vol_score = 25
            
            # Growth Score (15%)
            if saturation < 20:
                growth_score = 100
            elif saturation < 50:
                growth_score = 75
            else:
                growth_score = 50
            
            # Risk-adjusted Score (15%)
            if sharpe < 0:
                risk_score = 25
            elif sharpe < 0.5:
                risk_score = 50
            elif sharpe < 1.0:
                risk_score = 75
            else:
                risk_score = 100
            
            # Cycle Score (15%)
            if distance_ath > -10:
                cycle_score = 50
            elif distance_ath > -30:
                cycle_score = 75
            else:
                cycle_score = 100
            
            composite = (pp_score * 0.25 + vol_score * 0.20 + growth_score * 0.15 + 
                        risk_score * 0.15 + cycle_score * 0.15 + 50 * 0.10)
            
            return round(composite, 3)
        
        def assign_class(row):
            score = row.get('PPP_Q_Composite_Score', 50)
            asset_category = row.get('Asset_Category', 'stock')
            
            if asset_category == 'crypto':
                if score >= 70:
                    return 'A_PRESERVER'
                elif score >= 50:
                    return 'B_PARTIAL'
                elif score >= 30:
                    return 'C_ERODER'
                else:
                    return 'D_DESTROYER'
            else:
                if score >= 65:
                    return 'A_PRESERVER'
                elif score >= 45:
                    return 'B_PARTIAL'
                elif score >= 25:
                    return 'C_ERODER'
                else:
                    return 'D_DESTROYER'
        
        pppq_df['PPP_Q_Composite_Score'] = pppq_df.apply(calculate_composite_score, axis=1)
        pppq_df['PPP_Q_Class'] = pppq_df.apply(assign_class, axis=1)
        
        # Fill missing values
        numeric_cols = pppq_df.select_dtypes(include=[np.number]).columns
        pppq_df[numeric_cols] = pppq_df[numeric_cols].fillna(0).round(3)
        
        # Time-based splits (NO LEAKAGE!)
        train_mask = (pppq_df['Date'] >= pd.to_datetime(config.TRAIN_START)) & \
                    (pppq_df['Date'] <= pd.to_datetime(config.TRAIN_END))
        val_mask = (pppq_df['Date'] >= pd.to_datetime(config.VAL_START)) & \
                  (pppq_df['Date'] <= pd.to_datetime(config.VAL_END))
        test_mask = (pppq_df['Date'] >= pd.to_datetime(config.TEST_START)) & \
                   (pppq_df['Date'] <= pd.to_datetime(config.TEST_END))
        
        train_df = pppq_df[train_mask].copy()
        val_df = pppq_df[val_mask].copy()
        test_df = pppq_df[test_mask].copy()
        
        # Save processed data
        train_df.to_csv(config.TRAIN_DATA, index=False)
        val_df.to_csv(config.VAL_DATA, index=False)
        test_df.to_csv(config.TEST_DATA, index=False)
        
        # Save feature metadata
        feature_cols = [col for col in pppq_df.columns 
                       if col not in ['Date', 'Asset', 'PPP_Q_Class', 'Asset_Category']]
        
        feature_metadata = {
            'features': feature_cols,
            'target': 'PPP_Q_Class',
            'classes': ['A_PRESERVER', 'B_PARTIAL', 'C_ERODER', 'D_DESTROYER'],
            'num_features': len(feature_cols),
            'assets': config.CORE_ASSETS
        }
        
        with open(config.FEATURES_PATH, 'w') as f:
            json.dump(feature_metadata, f, indent=2)
        
        logger.info(f"✅ Preprocessing complete:")
        logger.info(f"   Train: {len(train_df)} rows")
        logger.info(f"   Val:   {len(val_df)} rows")
        logger.info(f"   Test:  {len(test_df)} rows")
        logger.info(f"   Features: {len(feature_cols)}")
        
        return train_df, val_df, test_df
    
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        raise


# ============================================================================
# TASK 3: MODEL TRAINING
# ============================================================================

@task(
    name="train_models",
    retries=config.RETRY_ATTEMPTS,
    retry_delay_seconds=config.RETRY_DELAY_SECONDS,
    timeout_seconds=config.TIMEOUT_SECONDS
)
def train_models(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train LightGBM, XGBoost, and Random Forest models
    
    Returns:
        Dict with model metrics and artifacts
    """
    logger.info("="*80)
    logger.info("TASK 3: MODEL TRAINING")
    logger.info("="*80)
    
    try:
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, f1_score, classification_report
        )
        
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
        
        logger.info(f"📊 Training shapes: X={X_train.shape}, y={len(y_train_enc)}")
        
        # ================================================================
        # Train LightGBM
        # ================================================================
        logger.info("\n🚀 Training LightGBM...")
        start_time = datetime.now()
        
        lgb_train = lgb.Dataset(X_train, label=y_train_enc, feature_name=feature_cols)
        lgb_val = lgb.Dataset(X_val, label=y_val_enc, reference=lgb_train)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=100)
        ]
        
        lgbm_model = lgb.train(
            config.LGBM_PARAMS,
            lgb_train,
            num_boost_round=config.NUM_BOOST_ROUND,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )
        
        lgbm_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"   ✅ LightGBM trained in {lgbm_time:.1f}s (best iter: {lgbm_model.best_iteration})")
        
        # ================================================================
        # Train XGBoost
        # ================================================================
        logger.info("\n🚀 Training XGBoost...")
        start_time = datetime.now()
        
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=4,
            eval_metric='mlogloss',
            max_depth=7,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50
        )
        
        xgb_model.fit(
            X_train, y_train_enc,
            eval_set=[(X_val, y_val_enc)],
            verbose=False
        )
        
        xgb_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"   ✅ XGBoost trained in {xgb_time:.1f}s")
        
        # ================================================================
        # Train Random Forest
        # ================================================================
        logger.info("\n🚀 Training Random Forest...")
        start_time = datetime.now()
        
        rf_model = RandomForestClassifier(**config.RF_PARAMS)
        rf_model.fit(X_train, y_train_enc)
        
        rf_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"   ✅ Random Forest trained in {rf_time:.1f}s")
        
        # ================================================================
        # Evaluate Models
        # ================================================================
        logger.info("\n📊 Evaluating models on test set...")
        
        def evaluate_model(model, X, y_true, model_type='lgb'):
            if model_type == 'lgb':
                y_pred_proba = model.predict(X, num_iteration=model.best_iteration)
            else:
                y_pred_proba = model.predict_proba(X)
            
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'macro_f1': f1_score(y_true, y_pred, average='macro'),
                'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        
        lgbm_metrics = evaluate_model(lgbm_model, X_test, y_test_enc, 'lgb')
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test_enc, 'xgb')
        rf_metrics = evaluate_model(rf_model, X_test, y_test_enc, 'rf')
        
        # Ensemble
        ensemble_proba = (lgbm_metrics['probabilities'] + xgb_metrics['probabilities']) / 2
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        ensemble_metrics = {
            'accuracy': accuracy_score(y_test_enc, ensemble_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test_enc, ensemble_pred),
            'macro_f1': f1_score(y_test_enc, ensemble_pred, average='macro'),
            'weighted_f1': f1_score(y_test_enc, ensemble_pred, average='weighted')
        }
        
        logger.info("\n📊 TEST SET RESULTS:")
        logger.info(f"   {'Model':<20} {'Macro F1':<12} {'Accuracy':<12}")
        logger.info(f"   {'-'*44}")
        logger.info(f"   {'LightGBM':<20} {lgbm_metrics['macro_f1']:<12.4f} {lgbm_metrics['accuracy']:<12.4f}")
        logger.info(f"   {'XGBoost':<20} {xgb_metrics['macro_f1']:<12.4f} {xgb_metrics['accuracy']:<12.4f}")
        logger.info(f"   {'Random Forest':<20} {rf_metrics['macro_f1']:<12.4f} {rf_metrics['accuracy']:<12.4f}")
        logger.info(f"   {'Ensemble':<20} {ensemble_metrics['macro_f1']:<12.4f} {ensemble_metrics['accuracy']:<12.4f}")
        
        # ================================================================
        # Save Models
        # ================================================================
        logger.info("\n💾 Saving models...")
        
        lgbm_model.save_model(str(config.LGBM_MODEL))
        xgb_model.save_model(str(config.XGB_MODEL))
        
        with open(config.RF_MODEL, 'wb') as f:
            pickle.dump(rf_model, f)
        
        with open(config.ENCODER_PATH, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        with open(config.FEATURE_COLUMNS, 'w') as f:
            json.dump({'features': feature_cols}, f, indent=2)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': lgbm_model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(config.REPORTS_DIR / 'feature_importance.csv', index=False)
        
        logger.info("✅ Models saved successfully")
        
        results = {
            'lightgbm': {
                'metrics': {k: v for k, v in lgbm_metrics.items() if k not in ['predictions', 'probabilities']},
                'training_time': lgbm_time,
                'best_iteration': lgbm_model.best_iteration
            },
            'xgboost': {
                'metrics': {k: v for k, v in xgb_metrics.items() if k not in ['predictions', 'probabilities']},
                'training_time': xgb_time
            },
            'random_forest': {
                'metrics': {k: v for k, v in rf_metrics.items() if k not in ['predictions', 'probabilities']},
                'training_time': rf_time
            },
            'ensemble': ensemble_metrics,
            'best_model': 'LightGBM' if lgbm_metrics['macro_f1'] >= xgb_metrics['macro_f1'] else 'XGBoost',
            'feature_importance': importance_df.head(20).to_dict('records')
        }
        
        return results
    
    except Exception as e:
        logger.error(f"❌ Model training failed: {str(e)}")
        raise


# ============================================================================
# TASK 4: MODEL EVALUATION & VERSIONING
# ============================================================================

@task(
    name="evaluate_and_version",
    retries=config.RETRY_ATTEMPTS,
    retry_delay_seconds=config.RETRY_DELAY_SECONDS
)
def evaluate_and_version(train_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate trained model and register in model registry
    
    Returns:
        Dict with evaluation results and deployment decision
    """
    logger.info("="*80)
    logger.info("TASK 4: MODEL EVALUATION & VERSIONING")
    logger.info("="*80)
    
    try:
        # Get best model metrics
        best_model = train_results['best_model']
        best_metrics = train_results['lightgbm']['metrics'] if best_model == 'LightGBM' else train_results['xgboost']['metrics']
        
        logger.info(f"🏆 Best Model: {best_model}")
        logger.info(f"   Macro F1: {best_metrics['macro_f1']:.4f}")
        logger.info(f"   Accuracy: {best_metrics['accuracy']:.4f}")
        
        # Check if should deploy
        should_deploy, prev_metrics = registry.should_deploy(
            best_metrics,
            threshold=config.MIN_IMPROVEMENT_THRESHOLD
        )
        
        # Log to registry
        run_id = registry.log_model_training(
            model_name='PPP-Q-Classifier',
            model_type=best_model.lower(),
            train_metrics=train_results['lightgbm']['metrics'],
            val_metrics={},  # Val metrics computed during training
            test_metrics=best_metrics,
            params=config.LGBM_PARAMS if best_model == 'LightGBM' else config.XGB_PARAMS,
            feature_importance={item['feature']: item['importance'] 
                               for item in train_results['feature_importance']}
        )
        
        logger.info(f"✅ Model logged: Run ID {run_id}")
        
        if should_deploy:
            registry.promote_to_staging(run_id)
            registry.promote_to_production(run_id)
            logger.info(f"✅ Model {run_id} promoted to Production")
        else:
            logger.warning("⚠️ Model not deployed (insufficient improvement)")
        
        # Save training summary
        summary = {
            'training_date': datetime.now().isoformat(),
            'run_id': run_id,
            'best_model': best_model,
            'deployed': should_deploy,
            'metrics': best_metrics,
            'previous_metrics': prev_metrics,
            'models': {
                'lightgbm': train_results['lightgbm'],
                'xgboost': train_results['xgboost'],
                'random_forest': train_results['random_forest'],
                'ensemble': train_results['ensemble']
            }
        }
        
        with open(config.REPORTS_DIR / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return {
            'run_id': run_id,
            'deployed': should_deploy,
            'best_model': best_model,
            'metrics': best_metrics,
            'previous_metrics': prev_metrics
        }
    
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        raise


# ============================================================================
# TASK 5: SEND NOTIFICATIONS
# ============================================================================

@task(
    name="send_notifications",
    retries=1
)
def send_notifications(
    pipeline_status: str,
    train_results: Dict[str, Any],
    eval_results: Dict[str, Any],
    new_rows: int,
    error_msg: Optional[str] = None
):
    """Send email notifications about pipeline execution"""
    logger.info("="*80)
    logger.info("TASK 5: SENDING NOTIFICATIONS")
    logger.info("="*80)
    
    try:
        if pipeline_status == "success":
            details = {
                'New Data Rows': f"{new_rows:,}",
                'Best Model': eval_results.get('best_model', 'N/A'),
                'Macro F1': f"{eval_results['metrics']['macro_f1']:.4f}",
                'Accuracy': f"{eval_results['metrics']['accuracy']:.4f}",
                'Balanced Accuracy': f"{eval_results['metrics']['balanced_accuracy']:.4f}",
                'Deployed': '✅ Yes' if eval_results.get('deployed') else '❌ No',
                'Run ID': eval_results.get('run_id', 'N/A'),
                'LightGBM F1': f"{train_results['lightgbm']['metrics']['macro_f1']:.4f}",
                'XGBoost F1': f"{train_results['xgboost']['metrics']['macro_f1']:.4f}",
                'Ensemble F1': f"{train_results['ensemble']['macro_f1']:.4f}"
            }
            
            notifier.notify_pipeline_success(
                pipeline_name="PPP-Q ML Pipeline",
                details=details
            )
            
            if eval_results.get('deployed'):
                notifier.notify_model_deployed(
                    model_version=eval_results.get('run_id', 'unknown'),
                    metrics=eval_results['metrics']
                )
        
        elif pipeline_status == "failure":
            notifier.notify_pipeline_failure(
                pipeline_name="PPP-Q ML Pipeline",
                error=error_msg or "Unknown error",
                details={'Time': datetime.now().isoformat()}
            )
        
        logger.info("✅ Notifications sent")
    
    except Exception as e:
        logger.error(f"❌ Notification failed: {str(e)}")


# ============================================================================
# MAIN FLOW
# ============================================================================

@flow(
    name="PPP-Q-ML-Pipeline",
    description="Complete ML pipeline: Data → Features → Train → Evaluate → Deploy",
    retries=0
)
def pppq_ml_pipeline():
    """
    Main Prefect flow orchestrating the entire ML pipeline
    
    Runs every 15 days automatically
    """
    pipeline_start = datetime.now()
    
    logger.info("="*80)
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*20 + "PPP-Q ML PIPELINE STARTED".center(48) + " "*10 + "║")
    logger.info("║" + f" Time: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}".ljust(78) + "║")
    logger.info("╚" + "="*78 + "╝")
    logger.info("="*80)
    
    # Notify start
    notifier.notify_pipeline_start("PPP-Q ML Pipeline")
    
    try:
        # TASK 1: Data Ingestion
        df_raw, new_rows = fetch_new_data()
        
        # Check minimum data
        if new_rows < config.MIN_DATA_SIZE and config.RAW_DATA_PATH.exists():
            logger.warning(f"⚠️ Only {new_rows} new rows (threshold: {config.MIN_DATA_SIZE})")
        
        # TASK 2: Preprocessing
        train_df, val_df, test_df = preprocess_data(df_raw)
        
        # TASK 3: Model Training
        train_results = train_models(train_df, val_df, test_df)
        
        # TASK 4: Evaluation & Versioning
        eval_results = evaluate_and_version(train_results)
        
        # TASK 5: Notifications
        send_notifications("success", train_results, eval_results, new_rows)
        
        # Pipeline complete
        duration = (datetime.now() - pipeline_start).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("╔" + "="*78 + "╗")
        logger.info("║" + " "*20 + "PIPELINE COMPLETED SUCCESSFULLY ✅".center(48) + " "*10 + "║")
        logger.info("║" + f" Duration: {duration:.1f} seconds".ljust(78) + "║")
        logger.info("║" + f" Best Model: {eval_results['best_model']} (F1: {eval_results['metrics']['macro_f1']:.4f})".ljust(78) + "║")
        logger.info("╚" + "="*78 + "╝")
        logger.info("="*80)
        
        return eval_results
    
    except Exception as e:
        logger.error(f"\n❌ PIPELINE FAILED: {str(e)}", exc_info=True)
        send_notifications("failure", {}, {}, 0, error_msg=str(e))
        raise


# ============================================================================
# ENTRY POINTS
# ============================================================================

def run_pipeline():
    """Run the pipeline (for manual execution)"""
    return pppq_ml_pipeline()


def schedule_pipeline():
    """Schedule pipeline to run every 15 days"""
    import schedule
    import time
    
    logger.info(f"📅 Scheduling pipeline to run every {config.PIPELINE_INTERVAL_DAYS} days")
    
    schedule.every(config.PIPELINE_INTERVAL_DAYS).days.do(pppq_ml_pipeline)
    
    # Also run immediately on first start
    pppq_ml_pipeline()
    
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour


if __name__ == "__main__":
    run_pipeline()
