"""
Fix PPP-Q Classification - Rebalanced Scoring System
"""

import pandas as pd
import numpy as np
import os

def calculate_pppq_composite_score(row):
    """
    Balanced PPP-Q Composite Score (0-100)
    
    Components:
    1. Purchasing Power (30%) - PP_Multiplier_5Y
    2. Volatility Risk (20%) - Volatility_90D (lower is better)
    3. Growth Potential (15%) - Market_Cap_Saturation_Pct (lower = more room)
    4. Risk-Adjusted (15%) - Sharpe_Ratio_5Y
    5. Market Cycle (10%) - Distance_From_ATH_Pct
    6. Consistency (10%) - Return_Consistency + PP_Stability_Index
    """
    asset_category = row.get('Asset_Category', 'stock')
    
    # 1. PP Score (30%) - Primary metric for purchasing power preservation
    pp_mult = row.get('PP_Multiplier_5Y', 1.0)
    if pp_mult >= 3.0:
        pp_score = 100
    elif pp_mult >= 2.0:
        pp_score = 85
    elif pp_mult >= 1.5:
        pp_score = 70
    elif pp_mult >= 1.2:
        pp_score = 55
    elif pp_mult >= 1.0:
        pp_score = 40
    elif pp_mult >= 0.8:
        pp_score = 25
    else:
        pp_score = 10
    
    # 2. Volatility Score (20%) - Asset-class adjusted
    volatility = row.get('Volatility_90D', 30)
    if asset_category == 'crypto':
        # Crypto: expect 30-80% volatility
        if volatility < 30:
            vol_score = 100
        elif volatility < 50:
            vol_score = 75
        elif volatility < 70:
            vol_score = 50
        else:
            vol_score = 25
    else:
        # Traditional: expect 10-30% volatility
        if volatility < 10:
            vol_score = 100
        elif volatility < 20:
            vol_score = 75
        elif volatility < 30:
            vol_score = 50
        else:
            vol_score = 25
    
    # 3. Growth Potential (15%) - Lower saturation = more growth
    saturation = row.get('Market_Cap_Saturation_Pct', 50)
    if saturation < 10:
        growth_score = 100
    elif saturation < 30:
        growth_score = 80
    elif saturation < 50:
        growth_score = 60
    elif saturation < 70:
        growth_score = 40
    else:
        growth_score = 20
    
    # 4. Risk-Adjusted Score (15%)
    sharpe = row.get('Sharpe_Ratio_5Y', 0)
    if sharpe >= 1.5:
        risk_score = 100
    elif sharpe >= 1.0:
        risk_score = 80
    elif sharpe >= 0.5:
        risk_score = 60
    elif sharpe >= 0:
        risk_score = 40
    else:
        risk_score = 20
    
    # 5. Cycle Score (10%) - Entry timing
    distance_ath = row.get('Distance_From_ATH_Pct', 0)
    if distance_ath > -5:  # Near ATH
        cycle_score = 30
    elif distance_ath > -15:
        cycle_score = 50
    elif distance_ath > -30:
        cycle_score = 70
    elif distance_ath > -50:
        cycle_score = 85
    else:
        cycle_score = 100  # Deep value
    
    # 6. Consistency (10%)
    stability = row.get('PP_Stability_Index', 0.5)
    consistency = row.get('Return_Consistency', 0.5)
    consistency_score = (stability + consistency) * 50
    
    # Weighted composite
    composite = (
        pp_score * 0.30 +
        vol_score * 0.20 +
        growth_score * 0.15 +
        risk_score * 0.15 +
        cycle_score * 0.10 +
        consistency_score * 0.10
    )
    
    return round(composite, 2)


def assign_pppq_class(row):
    """
    Assign class based on composite score
    
    A_PRESERVER: score >= 65 (Strong PP preservation + growth)
    B_PARTIAL: score >= 55 (Adequate PP preservation)
    C_ERODER: score >= 42 (Marginal, may lose to inflation)
    D_DESTROYER: score < 42 (Significant PP destruction)
    """
    score = row.get('PPP_Q_Composite_Score', 50)
    
    # Simple threshold-based classification
    if score >= 65:
        return 'A_PRESERVER'
    elif score >= 55:
        return 'B_PARTIAL'
    elif score >= 42:
        return 'C_ERODER'
    else:
        return 'D_DESTROYER'


def main():
    print("=" * 60)
    print("PPP-Q Classification Fix - Rebalanced Scoring")
    print("=" * 60)
    
    # Load all splits
    train_df = pd.read_csv('data/processed/pppq/train/pppq_train.csv')
    val_df = pd.read_csv('data/processed/pppq/val/pppq_val.csv')
    test_df = pd.read_csv('data/processed/pppq/test/pppq_test.csv')
    
    print(f"\nLoaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Apply scoring to all splits
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        df['PPP_Q_Composite_Score'] = df.apply(calculate_pppq_composite_score, axis=1)
        df['PPP_Q_Class'] = df.apply(assign_pppq_class, axis=1)
    
    # Show class distribution
    print("\n" + "=" * 60)
    print("Class Distribution (Rebalanced)")
    print("=" * 60)
    
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{name}:")
        dist = df['PPP_Q_Class'].value_counts()
        total = len(df)
        for cls in ['A_PRESERVER', 'B_PARTIAL', 'C_ERODER', 'D_DESTROYER']:
            count = dist.get(cls, 0)
            pct = (count / total) * 100
            print(f"  {cls}: {count:,} ({pct:.1f}%)")
    
    # Show sample classifications
    print("\n" + "=" * 60)
    print("Sample Classifications (Latest)")
    print("=" * 60)
    
    assets = ['Bitcoin', 'Ethereum', 'Gold','Gold_ETF','RealEstate_ETF', 'Silver', 'SP500', 'NASDAQ', 
              'Apple', 'Microsoft', 'Oil', 'TreasuryBond_ETF', 'DowJones', 'JPMorgan','Litecoin']
    
    for asset in assets:
        asset_data = test_df[test_df['Asset'] == asset]
        if len(asset_data) > 0:
            latest = asset_data.tail(1).iloc[0]
            pp = latest['PP_Multiplier_5Y']
            score = latest['PPP_Q_Composite_Score']
            cls = latest['PPP_Q_Class']
            vol = latest['Volatility_90D']
            print(f"  {asset:20s} | PP={pp:5.2f} | Vol={vol:5.1f}% | Score={score:5.1f} | {cls}")
    
    # Save updated splits
    train_df.to_csv('data/processed/pppq/train/pppq_train.csv', index=False)
    val_df.to_csv('data/processed/pppq/val/pppq_val.csv', index=False)
    test_df.to_csv('data/processed/pppq/test/pppq_test.csv', index=False)
    
    print("\n" + "=" * 60)
    print("✓ Updated data saved!")
    print("=" * 60)
    
    return train_df, val_df, test_df


if __name__ == '__main__':
    main()
