"""
================================================================================
PPP-Q CLASSIFICATION - ENHANCED PRODUCTION PREPROCESSING
================================================================================
Purchasing Power Preservation Quality (PPP-Q) Classification System

Purpose: Multi-asset investment strategy classifier with cycle awareness
Output: A/B/C/D with actionable insights (volatility, cycle, entry signals)

Author: Bilal Ahmad Sheikh
Institution: GIKI
Date: December 2024
================================================================================
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class PPPQConfig:
    """Enhanced configuration with asset-specific logic"""
    
    # Paths
    DATA_DIR = 'data/'
    RAW_DATA_PATH = 'data/raw/final_consolidated_dataset.csv'
    PROCESSED_DIR = 'data/processed/pppq/'
    
    # Time splits (NO LEAKAGE!)
    TRAIN_START = '2010-01-01'
    TRAIN_END = '2021-12-31'
    VAL_START = '2022-01-01'
    VAL_END = '2023-12-31'
    TEST_START = '2024-01-01'
    TEST_END = '2025-12-31'
    
    # Asset categories
    CRYPTO_ASSETS = ['Bitcoin', 'Ethereum', 'Litecoin']
    PRECIOUS_METALS = ['Gold', 'Silver']
    EQUITY_INDICES = ['SP500', 'NASDAQ', 'DowJones']
    COMMODITIES = ['Oil']
    ETFS = ['Gold_ETF', 'TreasuryBond_ETF', 'RealEstate_ETF']
    TECH_STOCKS = ['Apple', 'Microsoft', 'JPMorgan']
    
    CORE_ASSETS = (CRYPTO_ASSETS + PRECIOUS_METALS + EQUITY_INDICES + 
                   COMMODITIES + ETFS + TECH_STOCKS)
    
    # Asset-specific volatility penalties
    VOLATILITY_MULTIPLIERS = {
        'crypto': 2.0,      # Crypto gets 2x penalty
        'commodity': 1.5,   # Commodities 1.5x penalty
        'stock': 1.2,       # Stocks 1.2x penalty
        'metal': 0.8,       # Metals get bonus (stability valued)
        'etf': 1.0,         # ETFs normal
        'index': 1.0        # Indices normal
    }

config = PPPQConfig()

# Create directories
os.makedirs(config.PROCESSED_DIR, exist_ok=True)
for folder in ['train', 'val', 'test']:
    os.makedirs(config.PROCESSED_DIR + folder, exist_ok=True)

print("="*80)
print("  PPP-Q ENHANCED CLASSIFICATION - PRODUCTION PREPROCESSING")
print("="*80)

# ============================================================================
# 1. LOAD & VALIDATE DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING & VALIDATING DATA")
print("="*80)

df = pd.read_csv(config.RAW_DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"\n✅ Data loaded: {df.shape}")
print(f"   Date Range: {df['Date'].min()} to {df['Date'].max()}")
print(f"   Duration: {(df['Date'].max() - df['Date'].min()).days / 365:.1f} years")

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def round_to_3dp(value):
    """Round to 3 decimal places"""
    if pd.isna(value):
        return 0.0
    return round(float(value), 3)

def get_asset_category(asset):
    """Determine asset category for specific rules"""
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

def calculate_cycle_position(price_series, ma_200):
    """Calculate where asset is in market cycle"""
    current_price = price_series.iloc[-1] if len(price_series) > 0 else 0
    ma_200_val = ma_200.iloc[-1] if len(ma_200) > 0 else current_price
    
    if ma_200_val == 0:
        return 0.0
    
    distance_from_ma = round(((current_price - ma_200_val) / ma_200_val) * 100, 3)
    return distance_from_ma

def calculate_ath_distance(price_series):
    """Calculate distance from all-time high"""
    if len(price_series) == 0:
        return 0.0
    
    ath = price_series.max()
    current = price_series.iloc[-1]
    
    if ath == 0:
        return 0.0
    
    distance = round(((current - ath) / ath) * 100, 3)
    return distance

# ============================================================================
# 3. ENGINEER ENHANCED FEATURES
# ============================================================================

print("\n" + "="*80)
print("STEP 2: ENGINEERING ENHANCED PPP-Q FEATURES")
print("="*80)

all_asset_data = []

for asset in config.CORE_ASSETS:
    print(f"\n   Processing: {asset}")
    
    asset_category = get_asset_category(asset)
    asset_df = df[['Date']].copy()
    asset_df['Asset'] = asset
    asset_df['Asset_Category'] = asset_category
    
    # ========================================================================
    # CORE FEATURES (From existing columns)
    # ========================================================================
    
    # Real Returns
    for horizon in ['3Y', '5Y', '10Y']:
        col = f'{asset}_Real_Return_{horizon}'
        if col in df.columns:
            asset_df[f'Real_Return_{horizon}'] = df[col].apply(round_to_3dp)
        else:
            asset_df[f'Real_Return_{horizon}'] = 0.0
    
    # PP Multipliers
    for horizon in ['1Y', '5Y', '10Y']:
        col = f'{asset}_PP_Multiplier_{horizon}'
        if col in df.columns:
            asset_df[f'PP_Multiplier_{horizon}'] = df[col].apply(round_to_3dp)
        else:
            asset_df[f'PP_Multiplier_{horizon}'] = 0.0
    
    # PP vs Cash
    col = f'{asset}_PP_vs_Cash_5Y'
    if col in df.columns:
        asset_df['PP_vs_Cash_5Y'] = df[col].apply(round_to_3dp)
    else:
        asset_df['PP_vs_Cash_5Y'] = asset_df['PP_Multiplier_5Y']
    
    # Volatility & Risk Metrics
    vol_col = f'{asset}_Volatility_90D'
    asset_df['Volatility_90D'] = df[vol_col].apply(round_to_3dp) if vol_col in df.columns else 0.0
    
    sharpe_col = f'{asset}_Sharpe_Ratio_5Y'
    asset_df['Sharpe_Ratio_5Y'] = df[sharpe_col].apply(round_to_3dp) if sharpe_col in df.columns else 0.0
    
    calmar_col = f'{asset}_Calmar_Ratio_5Y'
    asset_df['Calmar_Ratio_5Y'] = df[calmar_col].apply(round_to_3dp) if calmar_col in df.columns else 0.0
    
    # Drawdowns
    dd_col = f'{asset}_Max_Drawdown'
    asset_df['Max_Drawdown'] = df[dd_col].abs().apply(round_to_3dp) if dd_col in df.columns else 0.0
    
    # ========================================================================
    # MARKET CYCLE FEATURES (NEW - CRITICAL!)
    # ========================================================================
    
    price_col = f'{asset}_Price' if f'{asset}_Price' in df.columns else asset
    ma_50_col = f'{asset}_MA_50D'
    ma_200_col = f'{asset}_MA_200D'
    momentum_50_col = f'{asset}_Momentum_50D'
    momentum_200_col = f'{asset}_Momentum_200D'
    
    if price_col in df.columns:
        # Distance from moving averages
        if ma_200_col in df.columns:
            asset_df['Distance_From_MA_200D_Pct'] = (
                ((df[price_col] - df[ma_200_col]) / df[ma_200_col] * 100)
                .apply(round_to_3dp)
            )
        else:
            asset_df['Distance_From_MA_200D_Pct'] = 0.0
        
        if ma_50_col in df.columns:
            asset_df['Distance_From_MA_50D_Pct'] = (
                ((df[price_col] - df[ma_50_col]) / df[ma_50_col] * 100)
                .apply(round_to_3dp)
            )
        else:
            asset_df['Distance_From_MA_50D_Pct'] = 0.0
        
        # Distance from ATH
        ath_series = df[price_col].expanding().max()
        asset_df['Distance_From_ATH_Pct'] = (
            ((df[price_col] - ath_series) / ath_series * 100)
            .apply(round_to_3dp)
        )
        
        # Days since ATH
        is_ath = (df[price_col] == ath_series)
        asset_df['Days_Since_ATH'] = (~is_ath).groupby(is_ath.cumsum()).cumcount()
        
    else:
        asset_df['Distance_From_MA_200D_Pct'] = 0.0
        asset_df['Distance_From_MA_50D_Pct'] = 0.0
        asset_df['Distance_From_ATH_Pct'] = 0.0
        asset_df['Days_Since_ATH'] = 0
    
    # Momentum indicators
    if momentum_50_col in df.columns:
        asset_df['Momentum_50D'] = df[momentum_50_col].apply(round_to_3dp)
    else:
        asset_df['Momentum_50D'] = 0.0
    
    if momentum_200_col in df.columns:
        asset_df['Momentum_200D'] = df[momentum_200_col].apply(round_to_3dp)
    else:
        asset_df['Momentum_200D'] = 0.0
    
    # ========================================================================
    # GROWTH POTENTIAL FEATURES (From existing columns)
    # ========================================================================
    
    sat_col = f'{asset}_Market_Cap_Saturation_Pct'
    asset_df['Market_Cap_Saturation_Pct'] = (
        df[sat_col].apply(round_to_3dp) if sat_col in df.columns else 50.0
    )
    
    growth_col = f'{asset}_Growth_Potential_Multiplier'
    asset_df['Growth_Potential_Multiplier'] = (
        df[growth_col].apply(round_to_3dp) if growth_col in df.columns else 1.0
    )
    
    # ========================================================================
    # QUALITY SCORES (From existing columns)
    # ========================================================================
    
    composite_col = f'{asset}_Composite_Score_5Y'
    asset_df['Composite_Score_5Y'] = (
        df[composite_col].apply(round_to_3dp) if composite_col in df.columns else 50.0
    )
    
    vol_adj_col = f'{asset}_Vol_Adj_PP_Score_5Y'
    asset_df['Vol_Adj_PP_Score_5Y'] = (
        df[vol_adj_col].apply(round_to_3dp) if vol_adj_col in df.columns else 50.0
    )
    
    # ========================================================================
    # COMPARATIVE FEATURES
    # ========================================================================
    
    # Correlations
    for other_asset in ['SP500', 'Gold']:
        corr_col = f'{asset}_{other_asset}_Correlation'
        if corr_col in df.columns:
            asset_df[f'Correlation_With_{other_asset}'] = df[corr_col].apply(round_to_3dp)
        else:
            asset_df[f'Correlation_With_{other_asset}'] = 0.0
    
    # CPI correlation
    cpi_corr_col = f'{asset}_CPI_Correlation'
    asset_df['CPI_Correlation'] = (
        df[cpi_corr_col].apply(round_to_3dp) if cpi_corr_col in df.columns else 0.0
    )
    
    # ========================================================================
    # CRYPTO-SPECIFIC FEATURES
    # ========================================================================
    
    # ========================================================================
    # CRYPTO-SPECIFIC FEATURES
    # ========================================================================

    if asset_category == 'crypto':
        scarcity_col = f'{asset}_Scarcity_Score'
        asset_df['Scarcity_Score'] = (
            df[scarcity_col].apply(round_to_3dp) if scarcity_col in df.columns else 0.0
        )
        
        # Bitcoin 4-year cycle (halving cycle)
        if asset == 'Bitcoin':
            # Bitcoin halving dates: 2012, 2016, 2020, 2024
            halving_dates = pd.to_datetime(['2012-11-28', '2016-07-09', '2020-05-11', '2024-04-19'])
            
            days_since_halving = []
            for date in df['Date']:
                # Filter past halvings for this date
                past_halvings = halving_dates[halving_dates <= date]
                
                if len(past_halvings) > 0:
                    # Get the most recent halving
                    last_halving = past_halvings[-1]  # Use [-1] instead of .iloc[-1]
                    days = (date - last_halving).days
                    days_since_halving.append(days)
                else:
                    days_since_halving.append(0)
            
            asset_df['Days_Since_Bitcoin_Halving'] = days_since_halving
            asset_df['Bitcoin_Cycle_Year'] = (
                (pd.Series(days_since_halving) / 365).apply(lambda x: round(x % 4, 3))
            )
        else:
            asset_df['Days_Since_Bitcoin_Halving'] = 0
            asset_df['Bitcoin_Cycle_Year'] = 0.0
    else:
        asset_df['Scarcity_Score'] = 0.0
        asset_df['Days_Since_Bitcoin_Halving'] = 0
        asset_df['Bitcoin_Cycle_Year'] = 0.0
    
    # ========================================================================
    # REAL COMMODITY FEATURES
    # ========================================================================
    
    # Real PP Index
    if 'Real_PP_Index' in df.columns:
        asset_df['Real_PP_Index'] = df['Real_PP_Index'].apply(round_to_3dp)
    else:
        asset_df['Real_PP_Index'] = 100.0
    
    # Eggs/Milk purchasing power
    if 'Eggs_Per_100USD' in df.columns:
        asset_df['Eggs_Per_100USD'] = df['Eggs_Per_100USD'].apply(round_to_3dp)
    else:
        asset_df['Eggs_Per_100USD'] = 0.0
    
    if 'Milk_Gallons_Per_100USD' in df.columns:
        asset_df['Milk_Gallons_Per_100USD'] = df['Milk_Gallons_Per_100USD'].apply(round_to_3dp)
    else:
        asset_df['Milk_Gallons_Per_100USD'] = 0.0
    
    # Asset denominated in commodities
    egg_return_col = f'{asset}_Real_Return_Eggs_1Y'
    if egg_return_col in df.columns:
        asset_df['Real_Return_Eggs_1Y'] = df[egg_return_col].apply(round_to_3dp)
    else:
        asset_df['Real_Return_Eggs_1Y'] = 0.0
    
    milk_return_col = f'{asset}_Real_Return_Milk_1Y'
    if milk_return_col in df.columns:
        asset_df['Real_Return_Milk_1Y'] = df[milk_return_col].apply(round_to_3dp)
    else:
        asset_df['Real_Return_Milk_1Y'] = 0.0
    
    # ========================================================================
    # REGIME FEATURES
    # ========================================================================
    
    if 'Inflation_Regime' in df.columns:
        asset_df['Inflation_Regime'] = df['Inflation_Regime']
    else:
        asset_df['Inflation_Regime'] = 'Medium'
    
    if 'Composite_Recession_Risk' in df.columns:
        asset_df['Recession_Risk'] = df['Composite_Recession_Risk'].apply(round_to_3dp)
    else:
        asset_df['Recession_Risk'] = 0.0
    
    if 'Yield_Curve_Inverted' in df.columns:
        asset_df['Yield_Curve_Inverted'] = df['Yield_Curve_Inverted'].astype(int)
    else:
        asset_df['Yield_Curve_Inverted'] = 0
    
    # ========================================================================
    # CALCULATED STABILITY METRICS
    # ========================================================================
    
    # PP Stability (lower std = more stable)
    if 'PP_Multiplier_5Y' in asset_df.columns:
        rolling_std = asset_df['PP_Multiplier_5Y'].rolling(window=252, min_periods=1).std()
        asset_df['PP_Stability_Index'] = (1 / (1 + rolling_std)).apply(round_to_3dp)
    else:
        asset_df['PP_Stability_Index'] = 0.500
    
    # Return consistency (CV across horizons)
    returns = asset_df[['Real_Return_3Y', 'Real_Return_5Y', 'Real_Return_10Y']].values
    returns_std = np.nanstd(returns, axis=1)
    returns_mean = np.nanmean(returns, axis=1)
    cv = np.where(np.abs(returns_mean) > 0, returns_std / np.abs(returns_mean), 0)
    asset_df['Return_Consistency'] = np.clip((1 - cv), 0, 1).round(3)
    
    # Recovery strength
    asset_df['Recovery_Strength'] = (
        np.clip((1 / (1 + asset_df['Days_Since_ATH'] / 365)), 0, 1).round(3)
    )
    
    all_asset_data.append(asset_df)
    print(f"      ✅ {asset}: {asset_df.shape[1]} features | Category: {asset_category}")

# Combine all assets
print(f"\n🔗 Combining all asset data...")
pppq_df = pd.concat(all_asset_data, ignore_index=True)
print(f"   ✅ Combined shape: {pppq_df.shape}")

# ============================================================================
# 4. CALCULATE COMPOSITE SCORES
# ============================================================================

print("\n" + "="*80)
print("STEP 3: CALCULATING ASSET-SPECIFIC COMPOSITE SCORES")
print("="*80)

def calculate_enhanced_composite_score(row):
    """
    Enhanced composite scoring with asset-specific logic
    """
    
    asset_category = row['Asset_Category']
    
    # ========================================================================
    # COMPONENT 1: Real Consumption PP (25%)
    # ========================================================================
    
    pp_mult = row['PP_Multiplier_5Y']
    
    if pp_mult < 0.85:
        consumption_score = 0.0
    elif pp_mult < 1.0:
        consumption_score = 20.0 + (pp_mult - 0.85) / 0.15 * 30.0
    elif pp_mult < 1.3:
        consumption_score = 50.0 + (pp_mult - 1.0) / 0.3 * 30.0
    elif pp_mult < 2.0:
        consumption_score = 80.0 + (pp_mult - 1.3) / 0.7 * 15.0
    else:
        consumption_score = min(100.0, 95.0 + np.log10(pp_mult - 2.0 + 1) * 5.0)
    
    # ========================================================================
    # COMPONENT 2: Volatility Penalty (20%) - ASSET-SPECIFIC
    # ========================================================================
    
    volatility = row['Volatility_90D']
    vol_multiplier = config.VOLATILITY_MULTIPLIERS.get(asset_category, 1.0)
    adjusted_vol = volatility * vol_multiplier
    
    if adjusted_vol < 10:
        vol_score = 100.0
    elif adjusted_vol < 15:
        vol_score = 90.0
    elif adjusted_vol < 25:
        vol_score = 70.0
    elif adjusted_vol < 40:
        vol_score = 45.0
    elif adjusted_vol < 60:
        vol_score = 20.0
    else:
        vol_score = max(0.0, 10.0 - (adjusted_vol - 60) / 10.0)
    
    # ========================================================================
    # COMPONENT 3: Market Cycle Position (15%) - NEW!
    # ========================================================================
    
    distance_ath = row['Distance_From_ATH_Pct']
    distance_ma200 = row['Distance_From_MA_200D_Pct']
    
    # Near ATH = risky, far from ATH = opportunity
    if distance_ath > -5:  # Within 5% of ATH
        ath_score = 30.0  # Risky
    elif distance_ath > -20:
        ath_score = 60.0
    elif distance_ath > -50:
        ath_score = 85.0
    else:
        ath_score = 100.0  # Deep value
    
    # Above MA200 = uptrend, below = downtrend
    if distance_ma200 > 20:
        ma_score = 40.0  # Overextended
    elif distance_ma200 > 0:
        ma_score = 80.0  # Healthy uptrend
    elif distance_ma200 > -20:
        ma_score = 60.0  # Correction
    else:
        ma_score = 30.0  # Bear market
    
    cycle_score = (ath_score * 0.6 + ma_score * 0.4)
    
    # ========================================================================
    # COMPONENT 4: Growth Potential (15%)
    # ========================================================================
    
    saturation = row['Market_Cap_Saturation_Pct']
    
    if saturation < 10:
        growth_score = 100.0  # Huge upside
    elif saturation < 30:
        growth_score = 85.0
    elif saturation < 50:
        growth_score = 65.0
    elif saturation < 70:
        growth_score = 45.0
    elif saturation < 90:
        growth_score = 25.0
    else:
        growth_score = 10.0  # Saturated
    
    # ========================================================================
    # COMPONENT 5: Consistency (10%)
    # ========================================================================
    
    consistency = row['Return_Consistency'] * 50 + row['PP_Stability_Index'] * 50
    
    # ========================================================================
    # COMPONENT 6: Recovery (10%)
    # ========================================================================
    
    max_dd = row['Max_Drawdown']
    recovery = row['Recovery_Strength']
    
    if max_dd < 10:
        dd_score = 60.0
    elif max_dd < 25:
        dd_score = 50.0
    elif max_dd < 50:
        dd_score = 35.0
    elif max_dd < 75:
        dd_score = 15.0
    else:
        dd_score = 0.0
    
    recovery_score = dd_score * 0.6 + recovery * 100 * 0.4
    
    # ========================================================================
    # COMPONENT 7: Risk-Adjusted (5%)
    # ========================================================================
    
    sharpe = row['Sharpe_Ratio_5Y']
    
    if sharpe < 0:
        risk_adj_score = 0.0
    elif sharpe < 0.5:
        risk_adj_score = sharpe / 0.5 * 50.0
    elif sharpe < 1.0:
        risk_adj_score = 50.0 + (sharpe - 0.5) / 0.5 * 30.0
    else:
        risk_adj_score = min(100.0, 80.0 + sharpe * 10.0)
    
    # ========================================================================
    # CRYPTO-SPECIFIC ADJUSTMENTS
    # ========================================================================
    
    if asset_category == 'crypto':
        # Bitcoin cycle penalty
        if row['Asset'] == 'Bitcoin':
            cycle_year = row['Bitcoin_Cycle_Year']
            # Year 0-1 post-halving: Bullish (bonus)
            # Year 2: Very bullish (bonus)
            # Year 3: Bearish (penalty)
            if cycle_year < 1.5:
                cycle_bonus = 10.0
            elif cycle_year < 2.5:
                cycle_bonus = 5.0
            else:
                cycle_bonus = -15.0  # Penalty for year 3
            
            cycle_score += cycle_bonus
            cycle_score = np.clip(cycle_score, 0, 100)
    
    # ========================================================================
    # FINAL COMPOSITE
    # ========================================================================
    
    composite = (
        consumption_score * 0.25 +
        vol_score * 0.20 +
        cycle_score * 0.15 +
        growth_score * 0.15 +
        consistency * 0.10 +
        recovery_score * 0.10 +
        risk_adj_score * 0.05
    )
    
    return round(composite, 3)

print("\n🔨 Calculating enhanced composite scores...")
pppq_df['PPP_Q_Composite_Score'] = pppq_df.apply(calculate_enhanced_composite_score, axis=1)

# ========================================================================
# ASSIGN CLASSES WITH ASSET-SPECIFIC LOGIC
# ========================================================================

def assign_pppq_class(row):
    """Asset-specific class assignment"""
    
    score = row['PPP_Q_Composite_Score']
    asset_category = row['Asset_Category']
    
    # Base thresholds
    if asset_category == 'crypto':
        # Crypto needs higher scores due to volatility
        if score >= 75:
            return 'A_PRESERVER'
        elif score >= 60:
            return 'B_PARTIAL'
        elif score >= 40:
            return 'C_ERODER'
        else:
            return 'D_DESTROYER'
    
    elif asset_category == 'metal':
        # Metals: stability rewarded
        if score >= 65:
            return 'A_PRESERVER'
        elif score >= 50:
            return 'B_PARTIAL'
        elif score >= 35:
            return 'C_ERODER'
        else:
            return 'D_DESTROYER'
    
    else:
        # Standard thresholds for others
        if score >= 70:
            return 'A_PRESERVER'
        elif score >= 55:
            return 'B_PARTIAL'
        elif score >= 35:
            return 'C_ERODER'
        else:
            return 'D_DESTROYER'

pppq_df['PPP_Q_Class'] = pppq_df.apply(assign_pppq_class, axis=1)

# Show results
print(f"\n📊 Class Distribution:")
print(pppq_df['PPP_Q_Class'].value_counts())

print(f"\n📊 Asset Distribution by Class:")
asset_dist = pd.crosstab(pppq_df['Asset'], pppq_df['PPP_Q_Class'], normalize='index') * 100
print(asset_dist.round(1))

# ============================================================================
# 5. HANDLE MISSING VALUES
# ============================================================================

print("\n" + "="*80)
print("STEP 4: FINAL DATA CLEANING")
print("="*80)

# Fill remaining NaN
numeric_cols = pppq_df.select_dtypes(include=[np.number]).columns
pppq_df[numeric_cols] = pppq_df[numeric_cols].fillna(0).round(3)

print(f"✅ All values rounded to 3 decimal places")
print(f"✅ Missing values handled")

# ============================================================================
# 6. TIME-BASED SPLIT
# ============================================================================

print("\n" + "="*80)
print("STEP 5: TIME-BASED TRAIN/VAL/TEST SPLIT")
print("="*80)

train_mask = (pppq_df['Date'] >= pd.to_datetime(config.TRAIN_START)) & (pppq_df['Date'] <= pd.to_datetime(config.TRAIN_END))
val_mask = (pppq_df['Date'] >= pd.to_datetime(config.VAL_START)) & (pppq_df['Date'] <= pd.to_datetime(config.VAL_END))
test_mask = (pppq_df['Date'] >= pd.to_datetime(config.TEST_START)) & (pppq_df['Date'] <= pd.to_datetime(config.TEST_END))

train_df = pppq_df[train_mask].copy()
val_df = pppq_df[val_mask].copy()
test_df = pppq_df[test_mask].copy()

for name, df_split in [('TRAIN', train_df), ('VAL', val_df), ('TEST', test_df)]:
    print(f"\n📅 {name} SET:")
    print(f"   Rows: {len(df_split):,}")
    print(f"   Classes: {df_split['PPP_Q_Class'].value_counts().to_dict()}")

# ============================================================================
# 7. SAVE PROCESSED DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 6: SAVING PROCESSED DATA")
print("="*80)

train_df.to_csv(config.PROCESSED_DIR + 'train/pppq_train.csv', index=False)
val_df.to_csv(config.PROCESSED_DIR + 'val/pppq_val.csv', index=False)
test_df.to_csv(config.PROCESSED_DIR + 'test/pppq_test.csv', index=False)

# Save feature list
feature_cols = [col for col in pppq_df.columns if col not in ['Date', 'Asset', 'PPP_Q_Class', 'Inflation_Regime', 'Asset_Category']]

feature_metadata = {
    'features': feature_cols,
    'target': 'PPP_Q_Class',
    'classes': ['A_PRESERVER', 'B_PARTIAL', 'C_ERODER', 'D_DESTROYER'],
    'num_features': len(feature_cols),
    'assets': config.CORE_ASSETS,
    'asset_categories': {
        'crypto': config.CRYPTO_ASSETS,
        'metals': config.PRECIOUS_METALS,
        'indices': config.EQUITY_INDICES,
        'commodities': config.COMMODITIES,
        'etfs': config.ETFS,
        'stocks': config.TECH_STOCKS
    }
}

with open(config.PROCESSED_DIR + 'pppq_features.json', 'w') as f:
    json.dump(feature_metadata, f, indent=2)

print(f"\n✅ Saved all datasets")
print(f"✅ Total features: {len(feature_cols)}")

print("\n" + "="*80)
print("✅ ENHANCED PPP-Q PREPROCESSING COMPLETE!")
print("="*80)
print(f"\n🚀 Ready for model training with:")
print(f"   • Asset-specific scoring logic")
print(f"   • Market cycle awareness")
print(f"   • Growth potential analysis")
print(f"   • All values rounded to 3 decimal places")
print(f"   • Production-ready feature set")