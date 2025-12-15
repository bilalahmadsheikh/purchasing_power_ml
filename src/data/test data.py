import os
import json
import pandas as pd

print("="*80)
print("PREPROCESSING VERIFICATION")
print("="*80)

# Check directory structure
processed_dir = 'data/processed/'

print(f"\n📁 Checking: {processed_dir}")
print("-"*80)

# List all files
for root, dirs, files in os.walk(processed_dir):
    level = root.replace(processed_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        file_path = os.path.join(root, file)
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f'{subindent}{file} ({file_size:.2f} MB)')

# Load and display summary
print("\n" + "="*80)
print("PREPROCESSING SUMMARY")
print("="*80)

try:
    with open(processed_dir + 'preprocessing_summary.json', 'r') as f:
        summary = json.load(f)
    
    print(f"\n✅ Preprocessing Date: {summary['preprocessing_date']}")
    print(f"\n📊 Dataset Info:")
    print(f"   Original: {summary['original_shape']}")
    print(f"   Cleaned: {summary['cleaned_shape']}")
    print(f"   Dropped: {summary['columns_dropped']} columns")
    
    print(f"\n📈 Splits:")
    for split_name, split_info in summary['splits'].items():
        print(f"   {split_name.upper()}:")
        print(f"      Rows: {split_info['rows']:,}")
        print(f"      Percentage: {split_info['percentage']:.1f}%")
        print(f"      Dates: {split_info['date_range']}")
    
    print(f"\n🎯 ML Tasks:")
    for task_name, task_info in summary['tasks'].items():
        print(f"   {task_name}:")
        print(f"      Type: {task_info['type']}")
        if 'features' in task_info:
            print(f"      Features: {task_info['features']}")
        if 'num_assets' in task_info:
            print(f"      Assets: {task_info['num_assets']}")
    
    if 'purchasing_power_metrics' in summary:
        print(f"\n💰 Purchasing Power Metrics:")
        print(f"   CPI-Adjusted: {summary['purchasing_power_metrics']['cpi_adjusted']}")
        print(f"   Real Commodities: {len(summary['purchasing_power_metrics']['real_commodities'])}")
        print(f"   Real PP Features: {summary['purchasing_power_metrics']['real_pp_features_available']}")
        
except FileNotFoundError:
    print("⚠️  Summary file not found!")

# Load feature sets
print("\n" + "="*80)
print("FEATURE SETS")
print("="*80)

try:
    with open(processed_dir + 'feature_sets.json', 'r') as f:
        feature_sets = json.load(f)
    
    print(f"\n📋 Available Feature Sets:")
    for task, info in feature_sets.items():
        if task == 'config':
            continue
        if isinstance(info, dict) and 'type' in info:
            print(f"\n   {task.upper()}:")
            print(f"      Type: {info['type']}")
            if 'features' in info and isinstance(info['features'], list):
                print(f"      Features: {len(info['features'])}")
                print(f"      Sample: {info['features'][:3]}")
            if 'features_dict' in info:
                print(f"      Assets: {len(info['features_dict'])}")
                print(f"      Asset List: {list(info['features_dict'].keys())[:5]}")
            if 'description' in info:
                print(f"      Description: {info['description']}")
    
    # Check for real PP features
    if 'real_pp_features' in feature_sets:
        real_pp = feature_sets['real_pp_features']
        print(f"\n💰 Real Purchasing Power Features: {len(real_pp)}")
        print(f"   Sample: {real_pp[:5]}")
        
except FileNotFoundError:
    print("⚠️  Feature sets file not found!")

# Quick data quality check
print("\n" + "="*80)
print("DATA QUALITY CHECK")
print("="*80)

try:
    # Load train data
    train = pd.read_csv(processed_dir + 'train/train_full.csv')
    
    print(f"\n✅ Train Data Loaded:")
    print(f"   Shape: {train.shape}")
    print(f"   Columns: {len(train.columns)}")
    print(f"   Missing values: {train.isnull().sum().sum()}")
    print(f"   Memory: {train.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check key columns exist
    key_cols = [
        'Date',
        'CPI',
        'Inflation_Rate_YoY',
        'Bitcoin_PP_Multiplier_5Y',
        'Gold_PP_Multiplier_5Y',
        'Eggs_Per_100USD',
        'Real_PP_Index'
    ]
    
    print(f"\n🔑 Key Columns Check:")
    for col in key_cols:
        exists = "✅" if col in train.columns else "❌"
        print(f"   {exists} {col}")
    
    # Check if real PP features are present
    real_pp_cols = [col for col in train.columns if any(x in col for x in [
        'Eggs', 'Milk', 'Bread', 'Gas', 'Real_PP', 'USD_Purchasing'
    ])]
    print(f"\n🥚 Real Commodity Features Found: {len(real_pp_cols)}")
    if real_pp_cols:
        print(f"   Examples: {real_pp_cols[:5]}")
    
except Exception as e:
    print(f"⚠️  Error loading train data: {e}")

print("\n" + "="*80)
print("✅ VERIFICATION COMPLETE!")
print("="*80)