import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from fredapi import Fred
import requests
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

print("✅ All libraries imported successfully!")

# ========================================
# CONFIGURATION
# ========================================

# Date range (adjust as needed)
START_DATE = "2010-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# FRED API Key - Load from environment variable
FRED_API_KEY = os.getenv('FRED_API_KEY', '')

# Output directory - Use project-relative path
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = str(PROJECT_ROOT / "data")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/raw/", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/processed/", exist_ok=True)

print(f"✅ Output directory created: {OUTPUT_DIR}")

# Initialize FRED API
fred = None
if FRED_API_KEY:
    try:
        fred = Fred(api_key=FRED_API_KEY)
        print("FRED API initialized successfully.")
    except Exception as e:
        print(f"⚠️ FRED API key provided but failed to initialize: {e}. Falling back to pandas_datareader for some FRED calls.")
else:
    print("⚠️ No FRED API key provided. Falling back to pandas_datareader for some FRED calls.")

# ========================================
# 1. FETCH ECONOMIC DATA (FRED)
# ========================================

def fetch_economic_data():
    """
    Fetch CPI, inflation, interest rates, money supply from FRED
    """
    print("\n📊 Fetching Economic Data from FRED...")

    economic_data = {}
    series = {
        'CPI': 'CPIAUCSL',              # Consumer Price Index
        'Fed_Funds_Rate': 'DFF',        # Federal Funds Rate
        'M2_Money_Supply': 'M2SL',      # M2 Money Supply
        'GDP': 'GDP',                   # Gross Domestic Product
        'Unemployment': 'UNRATE',       # Unemployment Rate
        'Core_CPI': 'CPILFESL',         # Core CPI (ex food & energy)
        'Treasury_10Y': 'DGS10',        # 10-Year Treasury Rate
        'Treasury_2Y': 'DGS2',          # 2-Year Treasury Rate
        'Real_GDP': 'GDPC1',            # Real GDP
    }

    for name, series_id in series.items():
        try:
            if fred: # Try FRED API first
                data = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
                economic_data[name] = data
            else: # Fallback to pandas_datareader
                data = pdr.DataReader(series_id, 'fred', START_DATE, END_DATE)
                economic_data[name] = data.squeeze() # Squeeze to Series if it's a single column DataFrame
            print(f"  ✓ {name}: {len(economic_data[name])} records")
        except Exception as e:
            print(f"  ✗ {name}: {str(e)}")

    df_economic = pd.DataFrame(economic_data)
    df_economic.index.name = 'Date'
    df_economic = df_economic.reset_index()

    # Calculate derived metrics
    # Ensure monthly data for CPI and M2 are handled for pct_change(12)
    df_economic['Inflation_Rate_YoY'] = df_economic['CPI'].pct_change(12) * 100
    df_economic['M2_Growth_Rate'] = df_economic['M2_Money_Supply'].pct_change(12) * 100
    df_economic['Real_Interest_Rate'] = df_economic['Fed_Funds_Rate'] - df_economic['Inflation_Rate_YoY']

    df_economic.to_csv(f"{OUTPUT_DIR}/raw/economic_data.csv", index=False)
    print(f"\n✅ Economic data saved: {len(df_economic)} rows")
    return df_economic

# ========================================
# 2. FETCH ASSET PRICES (Stocks, Gold, Commodities, VIX)
# ========================================

def fetch_asset_and_vix_prices():
    """
    Fetch historical prices for stocks, gold, commodities, ETFs, and VIX
    """
    print("\n💰 Fetching Asset Prices and VIX from Yahoo Finance...")

    assets = {
        # Precious Metals
        'Gold': 'GC=F',              # Gold Futures
        'Silver': 'SI=F',            # Silver Futures

        # Stock Indices
        'SP500': '^GSPC',            # S&P 500
        'NASDAQ': '^IXIC',           # NASDAQ
        'DowJones': '^DJI',          # Dow Jones

        # Commodities
        'Oil': 'CL=F',               # Crude Oil
        'NaturalGas': 'NG=F',        # Natural Gas

        # ETFs
        'Gold_ETF': 'GLD',           # SPDR Gold Trust
        'TreasuryBond_ETF': 'TLT',   # 20+ Year Treasury Bond ETF
        'RealEstate_ETF': 'VNQ',     # Vanguard Real Estate ETF

        # Individual Stocks (examples)
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'JPMorgan': 'JPM',

        # Volatility Index
        'VIX': '^VIX',               # VIX Index
    }

    df_list = []

    for name, ticker in assets.items():
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if not data.empty and 'Close' in data.columns:
                df_temp = data[['Close']].copy()
                df_temp.columns = [name]
                df_temp = df_temp.reset_index()
                df_temp['Date'] = pd.to_datetime(df_temp['Date'])
                if not df_temp[name].dropna().empty:
                    df_list.append(df_temp)
                    print(f"  ✓ {name} ({ticker}): {len(df_temp)} records")
                else:
                    print(f"  ✗ {name} ({ticker}): 'Close' column is all NaN or empty for the period.")
            else:
                print(f"  ✗ {name} ({ticker}): No data returned or 'Close' column missing.")
        except Exception as e:
            print(f"  ✗ {name} ({ticker}): {str(e)}")

    df_assets = None
    if df_list:
        df_assets = df_list[0]
        for i in range(1, len(df_list)):
            df_assets = pd.merge(df_assets, df_list[i], on='Date', how='outer')
        df_assets = df_assets.sort_values('Date').ffill().bfill()
        df_assets.to_csv(f"{OUTPUT_DIR}/raw/asset_prices_and_vix.csv", index=False)
        print(f"\n✅ Asset prices and VIX saved: {len(df_assets)} rows, {len(df_assets.columns) - 1} assets")
    else:
        df_assets = pd.DataFrame(columns=['Date'])
        print("\n⚠️ No asset prices or VIX were successfully fetched. Returning empty DataFrame.")
    return df_assets

# ========================================
# 3. FETCH CRYPTOCURRENCY DATA (Yahoo Finance)
# ========================================

def fetch_crypto_data_yfinance():
    """
    Fetch cryptocurrency data using Yahoo Finance
    """
    print("\n₿ Fetching Cryptocurrency Data from Yahoo Finance...")

    cryptos = {
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD',
        'Litecoin': 'LTC-USD',
        'Bitcoin_Cash': 'BCH-USD',
        'Cardano': 'ADA-USD',
        'Solana': 'SOL-USD',
    }

    crypto_data = []
    for name, ticker in cryptos.items():
        try:
            print(f"  Fetching {name} ({ticker})...")
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if not data.empty and 'Close' in data.columns:
                df_temp = data[['Close', 'Volume']].copy()
                df_temp.columns = [f'{name}_Price', f'{name}_Volume']
                df_temp = df_temp.reset_index()
                df_temp['Date'] = pd.to_datetime(df_temp['Date'])
                crypto_data.append(df_temp)
                print(f"    ✓ {name}: {len(df_temp)} records")
            else:
                print(f"    ✗ {name}: No data available")
        except Exception as e:
            print(f"    ✗ {name}: {str(e)}")

    df_crypto = None
    if crypto_data:
        df_crypto = crypto_data[0]
        for i in range(1, len(crypto_data)):
            df_crypto = pd.merge(df_crypto, crypto_data[i], on='Date', how='outer')
        df_crypto.to_csv(f"{OUTPUT_DIR}/raw/crypto_prices.csv", index=False)
        print(f"\n✅ Crypto data fetched: {len(df_crypto)} rows, {len(cryptos)} cryptocurrencies")
    else:
        df_crypto = pd.DataFrame(columns=['Date'])
        print("\n❌ No crypto data fetched")
    return df_crypto

def fetch_crypto_supply_yfinance():
    """
    Fetch current supply metrics for cryptocurrencies using Yahoo Finance
    """
    print("\n📊 Fetching Cryptocurrency Supply Metrics...")
    cryptos = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'BCH-USD', 'ADA-USD', 'SOL-USD']
    supply_data = []

    for ticker in cryptos:
        try:
            crypto = yf.Ticker(ticker)
            info = crypto.info
            supply_data.append({
                'Asset': ticker.replace('-USD', ''),
                'Symbol': info.get('symbol', ticker),
                'Market_Cap': info.get('marketCap'),
                'Circulating_Supply': info.get('circulatingSupply'),
                'Total_Supply': info.get('totalSupply'),
                'Max_Supply': info.get('maxSupply'),
                '24h_Volume': info.get('volume24Hr'),
            })
            print(f"  ✓ {ticker}")
        except Exception as e:
            print(f"  ✗ {ticker}: {str(e)}")
    df_supply = pd.DataFrame(supply_data)
    df_supply.to_csv(f"{OUTPUT_DIR}/raw/crypto_supply.csv", index=False)
    print(f"\n✅ Supply data fetched: {len(df_supply)} assets")
    return df_supply

# ========================================
# 4. FETCH STOCK SUPPLY DATA (Shares Outstanding)
# ========================================

def fetch_stock_supply_data():
    """
    Fetch shares outstanding for major stocks
    """
    print("\n📊 Fetching Stock Supply Data...")
    stocks = ['AAPL', 'MSFT', 'JPM', 'GLD']
    supply_data = []
    for ticker in stocks:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            supply_data.append({
                'Ticker': ticker,
                'Shares_Outstanding': info.get('sharesOutstanding'),
                'Float_Shares': info.get('floatShares'),
                'Market_Cap': info.get('marketCap'),
            })
            print(f"  ✓ {ticker}")
        except Exception as e:
            print(f"  ✗ {ticker}: {str(e)}")
    df_stock_supply = pd.DataFrame(supply_data)
    df_stock_supply.to_csv(f"{OUTPUT_DIR}/raw/stock_supply.csv", index=False)
    print(f"✅ Stock supply data saved: {len(df_stock_supply)} stocks")
    return df_stock_supply

# ========================================
# 5. FETCH REAL-WORLD BASELINES (FRED)
# ========================================

def fetch_real_baselines():
    """
    Fetch real-world price data for commodities (eggs, milk, bread, gas)
    """
    print("\n🥚 Fetching Real-World Commodity Baselines from FRED...")

    commodity_series = {
        'Eggs_Price_Per_Dozen': 'APU0000708111',
        'Milk_Price_Per_Gallon': 'APU0000709112',
        'Bread_Price_Per_Lb': 'APU0000702111',
        'Gas_Price_Per_Gallon': 'APU000074714',
    }
    commodity_data = {}

    for name, series_id in commodity_series.items():
        try:
            if fred:
                data = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
            else:
                data = pdr.DataReader(series_id, 'fred', START_DATE, END_DATE)
                data = data.squeeze()
            commodity_data[name] = data
            print(f"  ✓ {name}: {len(data)} records")
        except Exception as e:
            print(f"  ✗ {name}: {str(e)}")

    df_commodities = pd.DataFrame(commodity_data)
    df_commodities.index.name = 'Date'
    df_commodities = df_commodities.reset_index()
    df_commodities['Date'] = pd.to_datetime(df_commodities['Date'])

    df_commodities.to_csv(f"{OUTPUT_DIR}/raw/real_world_baselines.csv", index=False)
    print(f"\n✅ Real-world commodity baselines saved: {len(df_commodities)} rows")
    return df_commodities

# ========================================
# 6. FETCH GLOBAL MARKET DATA (FRED)
# ========================================

def fetch_global_market_data():
    """
    Fetch real global market data (M2, GDP) from FRED
    """
    print("\n🌍 Fetching Global Market Data from FRED...")

    us_m2_df = pd.DataFrame(columns=['Date', 'Global_M2_Trillions'])
    world_gdp_df = pd.DataFrame(columns=['Date', 'World_GDP_Trillions'])

    try:
        # US M2 (FRED series 'WM2NS' is weekly, let's just use monthly 'M2SL' for consistency with economic data)
        if fred:
            us_m2 = fred.get_series('M2SL', observation_start=START_DATE, observation_end=END_DATE)
        else:
            us_m2 = pdr.DataReader('M2SL', 'fred', START_DATE, END_DATE)
            us_m2 = us_m2.squeeze()
        us_m2_df = us_m2.reset_index()
        us_m2_df.columns = ['Date', 'US_M2_Billions']
        us_m2_df['Date'] = pd.to_datetime(us_m2_df['Date'])
        us_m2_df['Global_M2_Trillions'] = (us_m2_df['US_M2_Billions'] * 5 / 1000).round(4) # Rough estimate: global M2 is ~5x US M2
        print("  ✓ Fetched Global M2 proxy")
    except Exception as e:
        print(f"  ⚠️ Could not fetch Global M2: {e}. Defaulting to estimate.")
        us_m2_df = pd.DataFrame({'Date': pd.to_datetime([START_DATE, END_DATE]),
                                 'Global_M2_Trillions': [50.0, 150.0]}) # Placeholder

    try:
        # World GDP
        if fred:
            world_gdp = fred.get_series('NYGDPMKTPCDWLD', observation_start=START_DATE, observation_end=END_DATE)
        else:
            world_gdp = pdr.DataReader('NYGDPMKTPCDWLD', 'fred', START_DATE, END_DATE)
            world_gdp = world_gdp.squeeze()
        world_gdp_df = world_gdp.reset_index()
        world_gdp_df.columns = ['Date', 'World_GDP_Raw']
        world_gdp_df['Date'] = pd.to_datetime(world_gdp_df['Date'])
        world_gdp_df['World_GDP_Trillions'] = (world_gdp_df['World_GDP_Raw'] / 1e12).round(4)
        print("  ✓ Fetched World GDP")
    except Exception as e:
        print(f"  ⚠️ Could not fetch World GDP: {e}. Defaulting to estimate.")
        world_gdp_df = pd.DataFrame({'Date': pd.to_datetime([START_DATE, END_DATE]),
                                     'World_GDP_Trillions': [60.0, 100.0]}) # Placeholder

    return us_m2_df[['Date', 'Global_M2_Trillions']], world_gdp_df[['Date', 'World_GDP_Trillions']]

# ========================================
# 7. MERGE ALL RAW DATA
# ========================================

def merge_all_raw_data(df_economic, df_assets_vix, df_crypto_prices, df_commodities, df_global_m2, df_global_gdp):
    """
    Merge all raw data sources into a single time-series dataset
    """
    print("\n🔗 Merging all raw data sources...")

    df_merged = df_economic.copy()
    df_merged['Date'] = pd.to_datetime(df_merged['Date'])

    # Merge asset prices & VIX
    df_assets_vix['Date'] = pd.to_datetime(df_assets_vix['Date'])
    df_merged = pd.merge(df_merged, df_assets_vix, on='Date', how='outer')

    # Merge crypto prices
    if df_crypto_prices is not None and not df_crypto_prices.empty:
        df_crypto_prices['Date'] = pd.to_datetime(df_crypto_prices['Date'])
        df_merged = pd.merge(df_merged, df_crypto_prices, on='Date', how='outer')

    # Merge real-world commodity baselines
    df_commodities['Date'] = pd.to_datetime(df_commodities['Date'])
    df_merged = pd.merge(df_merged, df_commodities, on='Date', how='left') # left merge as commodities are monthly/quarterly

    # Merge global M2 and GDP (these are annual/monthly, so left merge and ffill)
    df_global_m2['Date'] = pd.to_datetime(df_global_m2['Date'])
    df_global_gdp['Date'] = pd.to_datetime(df_global_gdp['Date'])
    df_merged = pd.merge(df_merged, df_global_m2, on='Date', how='left')
    df_merged = pd.merge(df_merged, df_global_gdp, on='Date', how='left')


    # Sort by date and forward fill everything
    df_merged = df_merged.sort_values('Date').ffill().bfill() # bfill for initial NaNs

    # Drop rows with too many missing values (early dates where data might be sparse)
    df_merged = df_merged.dropna(thresh=len(df_merged.columns) * 0.5)

    df_merged.to_csv(f"{OUTPUT_DIR}/raw/merged_raw_data.csv", index=False)
    print(f"✅ Merged raw data saved: {len(df_merged)} rows, {len(df_merged.columns)} columns")
    return df_merged

# ========================================
# 8. FEATURE ENGINEERING HELPER FUNCTIONS
# ========================================

def add_column_safe(df, col_name, values):
    """Only add if doesn't exist"""
    if col_name not in df.columns:
        df[col_name] = values
        return True
    return False

def calculate_all_returns_volatility_technicals(df, asset_configs):
    """
    Calculate returns, volatility, and technicals for all specified assets.
    """
    print("\n⚙️ Calculating Returns, Volatility, Technicals for All Assets...")
    
    for asset_name, config in asset_configs.items():
        price_col = config['price_col']

        if price_col not in df.columns:
            print(f"  ⚠️ Skipping {asset_name} - price column '{price_col}' not found.")
            continue

        print(f"  Processing {asset_name}...")

        # Returns (various horizons)
        horizons = {'1M': 21, '3M': 63, '6M': 126, '1Y': 252, '3Y': 252 * 3, '5Y': 252 * 5, '10Y': 252 * 10}
        for horizon_name, days in horizons.items():
            return_col = f'{asset_name}_Return_{horizon_name}'
            add_column_safe(df, return_col, (df[price_col].pct_change(days) * 100))

        # Real returns (inflation-adjusted)
        if 'Inflation_Rate_YoY' in df.columns:
            for horizon_name in ['1Y', '3Y', '5Y', '10Y']:
                return_col = f'{asset_name}_Return_{horizon_name}'
                real_return_col = f'{asset_name}_Real_Return_{horizon_name}'
                if return_col in df.columns:
                    years = int(horizon_name[0]) if 'Y' in horizon_name else 1
                    avg_inflation = df['Inflation_Rate_YoY'].rolling(days, min_periods=1).mean() if days else df['Inflation_Rate_YoY']
                    cumulative_inflation = ((1 + avg_inflation/100) ** years - 1) * 100
                    add_column_safe(df, real_return_col, (df[return_col] - cumulative_inflation))

        # Volatility (rolling standard deviation)
        windows = {'30D': 30, '90D': 90}
        returns_daily = df[price_col].pct_change()
        for window_name, window_days in windows.items():
            vol_col = f'{asset_name}_Volatility_{window_name}'
            add_column_safe(df, vol_col, (returns_daily.rolling(window_days).std() * np.sqrt(252) * 100))

        # Moving averages & Momentum
        for ma_days in [50, 200]:
            ma_col = f'{asset_name}_MA_{ma_days}D'
            add_column_safe(df, ma_col, df[price_col].rolling(ma_days).mean())
            momentum_col = f'{asset_name}_Momentum_{ma_days}D'
            if ma_col in df.columns:
                add_column_safe(df, momentum_col, ((df[price_col] - df[ma_col]) / df[ma_col] * 100))

        # Maximum drawdown
        max_dd_col = f'{asset_name}_Max_Drawdown'
        if max_dd_col not in df.columns:
            rolling_max = df[price_col].expanding().max()
            add_column_safe(df, max_dd_col, ((df[price_col] - rolling_max) / rolling_max * 100))
        
        # Sharpe Ratio (using 1Y real return and 90D volatility)
        sharpe_col = f'{asset_name}_Sharpe_Ratio'
        real_return_1y = f'{asset_name}_Real_Return_1Y'
        volatility_90d = f'{asset_name}_Volatility_90D'
        if all(c in df.columns for c in [real_return_1y, volatility_90d]):
            add_column_safe(df, sharpe_col, (df[real_return_1y] / (df[volatility_90d].replace(0, np.nan) + 0.0001))) # Avoid division by zero

        # CPI Correlation
        corr_col = f'{asset_name}_CPI_Correlation'
        if 'CPI' in df.columns and corr_col not in df.columns:
            add_column_safe(df, corr_col, df[price_col].pct_change().rolling(252).corr(df['CPI'].pct_change()))

    return df

def engineer_core_features(df):
    """
    Calculate core derived features like time features, inflation/monetary regimes,
    and basic inter-asset correlations.
    """
    print("\n⚙️ Engineering Core Features...")

    df = df.copy()

    # Time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Inflation regime features
    if 'Inflation_Rate_YoY' in df.columns:
        df['Inflation_Accelerating'] = (df['Inflation_Rate_YoY'] > df['Inflation_Rate_YoY'].shift(1)).astype(int)
        df['High_Inflation_Period'] = (df['Inflation_Rate_YoY'] > 4.0).astype(int)
        df['Low_Inflation_Period'] = (df['Inflation_Rate_YoY'] < 2.0).astype(int)

    # Monetary policy features
    if 'Fed_Funds_Rate' in df.columns:
        df['Fed_Hawkish'] = (df['Fed_Funds_Rate'] > df['Fed_Funds_Rate'].shift(1)).astype(int)
        if 'Inflation_Rate_YoY' in df.columns:
            df['Taylor_Rule_Rate'] = 2 + df['Inflation_Rate_YoY'] + 0.5 * (df['Inflation_Rate_YoY'] - 2)
            df['Taylor_Rule_Deviation'] = df['Fed_Funds_Rate'] - df['Taylor_Rule_Rate']

    # Inter-asset correlations (examples)
    if all(c in df.columns for c in ['Gold', 'Bitcoin_Price']):
        df['Gold_Bitcoin_Correlation'] = df['Gold'].pct_change().rolling(90).corr(df['Bitcoin_Price'].pct_change())
    if all(c in df.columns for c in ['Bitcoin_Price', 'Ethereum_Price']):
        df['Bitcoin_Ethereum_Correlation'] = df['Bitcoin_Price'].pct_change().rolling(90).corr(df['Ethereum_Price'].pct_change())
    if all(c in df.columns for c in ['Bitcoin_Price', 'SP500']):
        df['Bitcoin_SP500_Correlation'] = df['Bitcoin_Price'].pct_change().rolling(90).corr(df['SP500'].pct_change())

    print(f"✅ Core feature engineering complete: {len(df.columns)} total columns")
    return df

def add_purchasing_power_multipliers_and_baselines(df, asset_configs):
    """
    Calculates purchasing power multipliers for all assets and adds cash/commodity baselines.
    """
    print("\n📈 Adding Purchasing Power Multipliers and Baselines...")

    horizons = ['1Y', '3Y', '5Y', '10Y']
    
    # Calculate Cash Baseline
    for horizon_name in horizons:
        years = int(horizon_name[0]) if 'Y' in horizon_name else 1
        if 'Inflation_Rate_YoY' in df.columns:
            horizon_days = 252 * years
            avg_inflation = df['Inflation_Rate_YoY'].rolling(horizon_days, min_periods=1).mean()
            add_column_safe(df, f'Cash_PP_Multiplier_{horizon_name}', (1 / (1 + avg_inflation/100) ** years).clip(0.1, 1.5))

    for asset_name, config in asset_configs.items():
        if config['price_col'] not in df.columns:
            continue
        
        for horizon in horizons:
            real_return_col = f'{asset_name}_Real_Return_{horizon}'
            if real_return_col in df.columns:
                # PP Multiplier
                pp_mult = (1 + df[real_return_col] / 100).clip(0, 10)
                add_column_safe(df, f'{asset_name}_PP_Multiplier_{horizon}', pp_mult)

                # Asset vs Cash Comparison
                cash_mult_col = f'Cash_PP_Multiplier_{horizon}'
                if cash_mult_col in df.columns:
                    add_column_safe(df, f'{asset_name}_PP_vs_Cash_{horizon}', (pp_mult / df[cash_mult_col]).clip(0, 20))
                    add_column_safe(df, f'{asset_name}_PP_Advantage_{horizon}', (pp_mult - df[cash_mult_col]) * 100)

    # Commodity Baselines for assets
    if 'Eggs_Price_Per_Dozen' in df.columns and 'Milk_Price_Per_Gallon' in df.columns:
        base_eggs = df['Eggs_Price_Per_Dozen'].iloc[0] if not df['Eggs_Price_Per_Dozen'].empty else 1
        base_milk = df['Milk_Price_Per_Gallon'].iloc[0] if not df['Milk_Price_Per_Gallon'].empty else 1
        base_bread = df['Bread_Price_Per_Lb'].iloc[0] if not df['Bread_Price_Per_Lb'].empty else 1
        base_gas = df['Gas_Price_Per_Gallon'].iloc[0] if not df['Gas_Price_Per_Gallon'].empty else 1

        add_column_safe(df, 'Real_PP_Index', (0.25 * (df['Eggs_Price_Per_Dozen'] / base_eggs) +
                                             0.25 * (df['Milk_Price_Per_Gallon'] / base_milk) +
                                             0.20 * (df['Bread_Price_Per_Lb'] / base_bread) +
                                             0.30 * (df['Gas_Price_Per_Gallon'] / base_gas)) * 100)
        add_column_safe(df, 'USD_Purchasing_Power', 100 / df['Real_PP_Index'] * 100)
        add_column_safe(df, 'Eggs_Per_100USD', 100 / df['Eggs_Price_Per_Dozen'])
        add_column_safe(df, 'Milk_Gallons_Per_100USD', 100 / df['Milk_Price_Per_Gallon'])

        for asset_name in ['Gold', 'Bitcoin', 'SP500']:
            price_col = f'{asset_name}_Price' if asset_name == 'Bitcoin' else asset_name
            if price_col in df.columns:
                add_column_safe(df, f'{asset_name}_In_Eggs', df[price_col] / df['Eggs_Price_Per_Dozen'])
                add_column_safe(df, f'{asset_name}_Real_Return_Eggs_1Y', df[f'{asset_name}_In_Eggs'].pct_change(252) * 100)
                add_column_safe(df, f'{asset_name}_In_Milk', df[price_col] / df['Milk_Price_Per_Gallon'])
                add_column_safe(df, f'{asset_name}_Real_Return_Milk_1Y', df[f'{asset_name}_In_Milk'].pct_change(252) * 100)

    print("✅ Purchasing power multipliers and baselines added.")
    return df

def add_economic_rules(df):
    """Apply economic rules and generate related features."""
    print("\n📜 Applying Economic Rules...")

    # Yield Curve Rule
    if 'Treasury_10Y' in df.columns and 'Treasury_2Y' in df.columns:
        df['Yield_Curve_Spread'] = df['Treasury_10Y'] - df['Treasury_2Y']
        df['Yield_Curve_Inverted'] = (df['Yield_Curve_Spread'] < 0).astype(int)
        df['Recession_Risk_From_Yield_Curve'] = (df['Yield_Curve_Spread'] * -10).clip(0, 100)

    # Money Supply Rule
    if 'M2_Growth_Rate' in df.columns and 'GDP' in df.columns:
        df['GDP_Growth_Rate'] = df['GDP'].pct_change(4) * 100
        df['Predicted_Inflation_From_M2'] = (df['M2_Growth_Rate'] - df['GDP_Growth_Rate'] - 2).clip(0, 20)
        if 'Inflation_Rate_YoY' in df.columns:
            df['Inflation_Surprise'] = df['Inflation_Rate_YoY'] - df['Predicted_Inflation_From_M2']

    # Real Rate Gold Rule
    if 'Real_Interest_Rate' in df.columns and 'Gold' in df.columns:
        base_gold = 1800
        df['Gold_Fair_Value_From_Real_Rate'] = (base_gold * (1 - df['Real_Interest_Rate'] * 0.10)).clip(500, 5000)
        df['Gold_Mispricing'] = ((df['Gold'] - df['Gold_Fair_Value_From_Real_Rate']) / df['Gold_Fair_Value_From_Real_Rate'] * 100)
        df['Gold_Signal_From_Real_Rate'] = pd.cut(df['Gold_Mispricing'], bins=[-np.inf, -20, -10, 10, 20, np.inf],
                                                  labels=['Strong_Buy', 'Buy', 'Hold', 'Sell', 'Strong_Sell'])

    # Bitcoin Stock-to-Flow Rule
    if 'Bitcoin_Estimated_Supply' in df.columns and 'Bitcoin_Price' in df.columns:
        df['Bitcoin_Stock_to_Flow'] = df['Bitcoin_Estimated_Supply'] / (144 * 365.25 * 6.25)
        df['Bitcoin_Price_From_S2F'] = (0.4 * (df['Bitcoin_Stock_to_Flow'] ** 3)).clip(100, 1000000)
        df['Bitcoin_S2F_Mispricing'] = ((df['Bitcoin_Price'] - df['Bitcoin_Price_From_S2F']) / df['Bitcoin_Price_From_S2F'] * 100)
        df['Bitcoin_Signal_From_S2F'] = pd.cut(df['Bitcoin_S2F_Mispricing'], bins=[-np.inf, -50, -20, 20, 50, np.inf],
                                               labels=['Strong_Buy', 'Buy', 'Hold', 'Sell', 'Strong_Sell'])

    # Unemployment Recession Rule (Sahm Rule)
    if 'Unemployment' in df.columns:
        df['Unemployment_3M_Change'] = df['Unemployment'].diff(63)
        df['Unemployment_12M_Low'] = df['Unemployment'].rolling(252, min_periods=1).min()
        df['Unemployment_3M_Avg'] = df['Unemployment'].rolling(63, min_periods=1).mean()
        df['Sahm_Rule_Indicator'] = df['Unemployment_3M_Avg'] - df['Unemployment_12M_Low']
        df['Recession_Risk_From_Unemployment'] = (df['Sahm_Rule_Indicator'] > 0.5).astype(int)
        df['Recession_Risk_Score_Unemployment'] = (df['Sahm_Rule_Indicator'] / 2.0 * 100).clip(0, 100)

    # Composite Recession Risk
    recession_cols = ['Recession_Risk_From_Yield_Curve', 'Recession_Risk_Score_Unemployment']
    available_cols = [c for c in recession_cols if c in df.columns]
    if available_cols:
        df['Composite_Recession_Risk'] = df[available_cols].mean(axis=1)
        df['Recession_Risk_Level'] = pd.cut(df['Composite_Recession_Risk'], bins=[0, 20, 40, 60, 100],
                                            labels=['Low', 'Moderate', 'High', 'Very_High'])

    print("✅ Economic rules applied.")
    return df

def get_all_asset_configs(df, crypto_supplies):
    """
    Create a comprehensive configuration for all assets including dynamic global market data.
    """
    global_m2 = df['Global_M2_Trillions'].iloc[-1] if 'Global_M2_Trillions' in df.columns else 100.0
    world_gdp = df['World_GDP_Trillions'].iloc[-1] if 'World_GDP_Trillions' in df.columns else 100.0

    # Ensure crypto supplies are available, use placeholders if not
    for crypto_name in ['Bitcoin', 'Ethereum', 'Litecoin', 'Bitcoin_Cash', 'Cardano', 'Solana']:
        if crypto_name not in crypto_supplies:
            crypto_supplies[crypto_name] = {'circulating': 1e7, 'max': 1e8} # Default values

    configs = {
        'Bitcoin': {'price_col': 'Bitcoin_Price', 'category': 'crypto', 'supply': crypto_supplies['Bitcoin']['circulating'], 'tam_trillions': global_m2 * 0.10, 'asset_class': 'Digital Asset'},
        'Ethereum': {'price_col': 'Ethereum_Price', 'category': 'crypto', 'supply': crypto_supplies['Ethereum']['circulating'], 'tam_trillions': global_m2 * 0.05, 'asset_class': 'Digital Asset'},
        'Litecoin': {'price_col': 'Litecoin_Price', 'category': 'crypto', 'supply': crypto_supplies['Litecoin']['circulating'], 'tam_trillions': global_m2 * 0.01, 'asset_class': 'Digital Asset'},
        'Bitcoin_Cash': {'price_col': 'Bitcoin_Cash_Price', 'category': 'crypto', 'supply': crypto_supplies['Bitcoin_Cash']['circulating'], 'tam_trillions': global_m2 * 0.005, 'asset_class': 'Digital Asset'},
        'Cardano': {'price_col': 'Cardano_Price', 'category': 'crypto', 'supply': crypto_supplies['Cardano']['circulating'], 'tam_trillions': global_m2 * 0.02, 'asset_class': 'Digital Asset'},
        'Solana': {'price_col': 'Solana_Price', 'category': 'crypto', 'supply': crypto_supplies['Solana']['circulating'], 'tam_trillions': global_m2 * 0.01, 'asset_class': 'Digital Asset'},
        'Gold': {'price_col': 'Gold', 'category': 'commodity', 'supply': 208000 * 32150.7, 'tam_trillions': global_m2 * 0.15, 'asset_class': 'Precious Metal'},
        'Silver': {'price_col': 'Silver', 'category': 'commodity', 'supply': 1600000 * 32150.7, 'tam_trillions': global_m2 * 0.05, 'asset_class': 'Precious Metal'},
        'Oil': {'price_col': 'Oil', 'category': 'commodity', 'supply': None, 'tam_trillions': world_gdp * 0.05, 'asset_class': 'Energy Commodity', 'market_cap_proxy': 'revenue_based'},
        'NaturalGas': {'price_col': 'NaturalGas', 'category': 'commodity', 'supply': None, 'tam_trillions': world_gdp * 0.02, 'asset_class': 'Energy Commodity', 'market_cap_proxy': 'revenue_based'},
        'SP500': {'price_col': 'SP500', 'category': 'equity_index', 'supply': None, 'tam_trillions': world_gdp * 0.40, 'asset_class': 'US Large Cap Equity', 'market_cap_approx': 40000},
        'NASDAQ': {'price_col': 'NASDAQ', 'category': 'equity_index', 'supply': None, 'tam_trillions': world_gdp * 0.25, 'asset_class': 'US Tech Equity', 'market_cap_approx': 25000},
        'DowJones': {'price_col': 'DowJones', 'category': 'equity_index', 'supply': None, 'tam_trillions': world_gdp * 0.20, 'asset_class': 'US Blue Chip Equity', 'market_cap_approx': 10000},
        'Gold_ETF': {'price_col': 'Gold_ETF', 'category': 'etf', 'supply': None, 'tam_trillions': global_m2 * 0.02, 'asset_class': 'Gold ETF', 'tracks': 'Gold'},
        'TreasuryBond_ETF': {'price_col': 'TreasuryBond_ETF', 'category': 'etf', 'supply': None, 'tam_trillions': global_m2 * 0.30, 'asset_class': 'Fixed Income ETF', 'tracks': 'Treasury Bonds'},
        'RealEstate_ETF': {'price_col': 'RealEstate_ETF', 'category': 'etf', 'supply': None, 'tam_trillions': world_gdp * 0.15, 'asset_class': 'Real Estate ETF', 'tracks': 'REITs'},
        'Apple': {'price_col': 'Apple', 'category': 'stock', 'supply': None, 'tam_trillions': world_gdp * 0.10, 'asset_class': 'Tech Stock', 'market_cap_approx': 3000},
        'Microsoft': {'price_col': 'Microsoft', 'category': 'stock', 'supply': None, 'tam_trillions': world_gdp * 0.10, 'asset_class': 'Tech Stock', 'market_cap_approx': 3000},
        'JPMorgan': {'price_col': 'JPMorgan', 'category': 'stock', 'supply': None, 'tam_trillions': world_gdp * 0.05, 'asset_class': 'Financial Stock', 'market_cap_approx': 500}
    }
    return configs

def add_market_cap_saturation_and_risk(df, us_m2_df, world_gdp_df, crypto_supply_data):
    """Add market cap, TAM, saturation, risk tiers, and composite scores for all assets."""
    print("\n💰 Adding Market Cap, Saturation, and Risk Metrics...")

    # Merge global data (these are already ffilled/bfilled from raw merge, just ensure present)
    if 'Global_M2_Trillions' not in df.columns and us_m2_df is not None:
        df = df.merge(us_m2_df, on='Date', how='left').ffill().bfill()
    if 'World_GDP_Trillions' not in df.columns and world_gdp_df is not None:
        df = df.merge(world_gdp_df, on='Date', how='left').ffill().bfill()

    crypto_supplies = {row['Asset']: {'circulating': row['Circulating_Supply'], 'max': row['Max_Supply']}
                       for idx, row in crypto_supply_data.iterrows()}

    asset_configs = get_all_asset_configs(df, crypto_supplies)

    for asset, config in asset_configs.items():
        price_col = config['price_col']

        if price_col not in df.columns:
            print(f"  ⚠️ Skipping {asset} (price column not found).")
            continue

        print(f"  Processing market cap for {asset}...")

        # Calculate market cap
        market_cap_billions = pd.Series(np.nan, index=df.index)
        if config['supply'] is not None:
            market_cap_billions = (df[price_col] * config['supply'] / 1e9)
        elif 'market_cap_approx' in config:
            market_cap_billions = pd.Series([config['market_cap_approx']] * len(df), index=df.index)
        elif config.get('market_cap_proxy') == 'revenue_based':
            market_cap_billions = (df[price_col] * 100) # Arbitrary scaling for commodities
        else:
            market_cap_billions = (df[price_col] * 10) # Rough proxy for ETFs/others

        add_column_safe(df, f'{asset}_Market_Cap_Billions', market_cap_billions)

        # TAM
        tam_trillions = pd.Series([config['tam_trillions']] * len(df), index=df.index)
        add_column_safe(df, f'{asset}_TAM_Trillions', tam_trillions)

        if f'{asset}_Market_Cap_Billions' in df.columns and f'{asset}_TAM_Trillions' in df.columns:
            saturation_pct = (df[f'{asset}_Market_Cap_Billions'] / (df[f'{asset}_TAM_Trillions'] * 1000) * 100).clip(0, 100)
            add_column_safe(df, f'{asset}_Market_Cap_Saturation_Pct', saturation_pct)
            add_column_safe(df, f'{asset}_Room_To_Grow', (100 - saturation_pct))
            add_column_safe(df, f'{asset}_Growth_Potential_Multiplier', (1 / (1 + saturation_pct / 100) ** 0.5))

        add_column_safe(df, f'{asset}_Asset_Class', config['asset_class'])

        # Bitcoin specific supply data
        if asset == 'Bitcoin':
            def estimate_btc_supply(date):
                genesis_date = pd.Timestamp('2009-01-03')
                days_since_genesis = (pd.Timestamp(date) - genesis_date).days
                blocks = days_since_genesis * 144
                reward_periods = [210000, 420000, 630000, 840000]
                rewards = [50, 25, 12.5, 6.25, 3.125]
                total_supply = 0
                current_blocks = 0
                for i, period_blocks in enumerate(reward_periods):
                    if blocks > period_blocks:
                        total_supply += (period_blocks - current_blocks) * rewards[i]
                        current_blocks = period_blocks
                    else:
                        total_supply += (blocks - current_blocks) * rewards[i]
                        break
                if blocks > reward_periods[-1]:
                    total_supply += (blocks - current_blocks) * rewards[len(reward_periods)]

                return min(total_supply, 21000000) # Cap at 21M BTC

            df['Bitcoin_Estimated_Supply'] = df['Date'].apply(estimate_btc_supply)
            df['Bitcoin_Supply_Remaining'] = 21000000 - df['Bitcoin_Estimated_Supply']
            df['Bitcoin_Scarcity_Score'] = (df['Bitcoin_Estimated_Supply'] / 21000000) * 100

    print("✅ Market cap and saturation metrics added.")

    # Risk Tiers and Composite Scores
    print("  Adding Risk Tiers & Composite Scores...")
    horizons = ['1Y', '3Y', '5Y', '10Y']

    for asset, config in asset_configs.items():
        if config['price_col'] not in df.columns:
            continue
        
        vol_col = f'{asset}_Volatility_90D'
        if vol_col in df.columns:
            add_column_safe(df, f'{asset}_Risk_Tier', pd.cut(df[vol_col], bins=[0, 15, 30, 60, 1000], labels=['Low_Risk', 'Medium_Risk', 'High_Risk', 'Very_High_Risk']))
        
        for horizon in horizons:
            real_return_col = f'{asset}_Real_Return_{horizon}'
            pp_mult_col = f'{asset}_PP_Multiplier_{horizon}'
            max_dd_col = f'{asset}_Max_Drawdown'

            if real_return_col in df.columns and vol_col in df.columns:
                add_column_safe(df, f'{asset}_Sharpe_Ratio_{horizon}', (df[real_return_col] / (df[vol_col].replace(0, np.nan) + 0.01)))
                if max_dd_col in df.columns:
                    add_column_safe(df, f'{asset}_Calmar_Ratio_{horizon}', (df[real_return_col] / (abs(df[max_dd_col]).replace(0, np.nan) + 1.0)))

            if pp_mult_col in df.columns and vol_col in df.columns:
                vol_penalty = (1 - (df[vol_col] / 200).clip(0, 0.5))
                add_column_safe(df, f'{asset}_Vol_Adj_PP_Multiplier_{horizon}', (df[pp_mult_col] * vol_penalty))
                add_column_safe(df, f'{asset}_Vol_Adj_PP_Score_{horizon}', ((df[f'{asset}_Vol_Adj_PP_Multiplier_{horizon}'] - 0.5) / 2.0 * 100).clip(0, 100))

        # Composite scores (5Y focus)
        pp_mult_5y = f'{asset}_PP_Multiplier_5Y'
        growth_pot = f'{asset}_Growth_Potential_Multiplier'
        vol = f'{asset}_Volatility_90D'

        if all(c in df.columns for c in [pp_mult_5y, growth_pot, vol]):
            vol_factor = (1 / (1 + df[vol] / 50))
            composite = (df[pp_mult_5y] * df[growth_pot] * vol_factor * 100).clip(0, 200)
            add_column_safe(df, f'{asset}_Composite_Score_5Y', composite)
            add_column_safe(df, f'{asset}_Final_Recommendation_5Y', pd.cut(composite, bins=[0, 60, 100, 130, 200], labels=['Poor', 'Moderate', 'Good', 'Excellent']))
    
    # Update best asset composite
    composite_cols = [c for c in df.columns if 'Composite_Score_5Y' in c and not 'Best_Asset_Composite_5Y' in c]
    if composite_cols:
        best = df[composite_cols].idxmax(axis=1).str.replace('_Composite_Score_5Y', '')
        add_column_safe(df, 'Best_Asset_Composite_5Y', best)

    print("✅ Risk tiers and composite scores added.")
    return df

# ========================================
# 9. CREATE LABELS (Target Variables)
# ========================================

def create_labels(df, asset_configs):
    """
    Create target labels for ML training, including general and crypto-specific ones.
    """
    print("\n🎯 Creating Labels...")

    df = df.copy()
    
    # Label 1: Inflation Regime Classification
    if 'Inflation_Rate_YoY' in df.columns:
        add_column_safe(df, 'Inflation_Regime', pd.cut(df['Inflation_Rate_YoY'], bins=[-np.inf, 2.0, 4.0, np.inf], labels=['Low', 'Medium', 'High']))

    # Label 2: Asset Protection Scores & Recommendations (1Y focus, for selected assets)
    assets_for_protection_score = ['Gold', 'SP500', 'Bitcoin', 'Ethereum', 'Litecoin']
    
    for asset_name in assets_for_protection_score:
        real_return_col = f'{asset_name}_Real_Return_1Y'
        volatility_col = f'{asset_name}_Volatility_30D'
        cpi_corr_col = f'{asset_name}_CPI_Correlation'

        if all(col in df.columns for col in [real_return_col, volatility_col]):
            return_score = df[real_return_col].clip(
                -50 if 'crypto' in asset_configs.get(asset_name, {}).get('category', '') else -20,
                100 if 'crypto' in asset_configs.get(asset_name, {}).get('category', '') else 50
            )
            return_score = (return_score + (50 if 'crypto' in asset_configs.get(asset_name, {}).get('category', '') else 20)) / (150 if 'crypto' in asset_configs.get(asset_name, {}).get('category', '') else 70) * 50

            volatility_score = df[volatility_col].clip(0, 200)
            volatility_score = (200 - volatility_score) / 200 * 30

            hedge_score = 10
            if cpi_corr_col in df.columns:
                hedge_score = df[cpi_corr_col].clip(-1, 1)
                hedge_score = (hedge_score + 1) / 2 * 20
            
            protection_score = (return_score + volatility_score + hedge_score).clip(0, 100)
            add_column_safe(df, f'{asset_name}_Protection_Score', protection_score)
            add_column_safe(df, f'{asset_name}_Recommendation', pd.cut(protection_score, bins=[0, 40, 60, 75, 100], labels=['Poor', 'Moderate', 'Good', 'Excellent']))

            add_column_safe(df, f'{asset_name}_Preserved_Purchasing_Power', (df[real_return_col] > 0).astype(int))

    # Multi-horizon labels (for selected assets)
    horizons = ['1Y', '3Y', '5Y', '10Y']
    assets_for_horizon_labels = ['Gold', 'Bitcoin', 'SP500', 'Ethereum', 'Litecoin']

    for asset_name in assets_for_horizon_labels:
        for horizon in horizons:
            mult_col = f'{asset_name}_PP_Multiplier_{horizon}'
            if mult_col in df.columns:
                add_column_safe(df, f'{asset_name}_Preserved_PP_{horizon}', (df[mult_col] > 1.0).astype(int))
                add_column_safe(df, f'{asset_name}_PP_Category_{horizon}', pd.cut(df[mult_col], bins=[0, 0.85, 1.0, 1.15, 1.35, 100], labels=['Severe_Loss', 'Loss', 'Moderate_Gain', 'Good_Gain', 'Excellent_Gain']))
                add_column_safe(df, f'{asset_name}_PP_Score_{horizon}', ((df[mult_col] - 0.5) / 2.0 * 100).clip(0, 100))

    # Best Asset per horizon
    for horizon in horizons:
        mult_cols = [f'{asset}_PP_Multiplier_{horizon}' for asset in assets_for_horizon_labels if f'{asset}_PP_Multiplier_{horizon}' in df.columns]
        if mult_cols:
            add_column_safe(df, f'Best_Asset_{horizon}', df[mult_cols].idxmax(axis=1).str.replace(f'_PP_Multiplier_{horizon}', ''))

    print(f"✅ Labels created successfully")
    return df

# ========================================
# 10. ROUND ALL NUMERICAL COLUMNS
# ========================================

def round_all_numerical_columns(df, decimals=4):
    """
    Rounds all numerical columns in the DataFrame to the specified number of decimal places.
    """
    print(f"\n🔢 Rounding all numerical columns to {decimals} decimal places...")
    numerical_cols = df.select_dtypes(include=np.number).columns
    df[numerical_cols] = df[numerical_cols].round(decimals)
    print(f"✅ Rounded {len(numerical_cols)} numerical columns.")
    return df

# ========================================
# 11. MAIN ORCHESTRATION FUNCTION
# ========================================

def generate_consolidated_dataset():
    """
    Main function to orchestrate data collection, processing, and feature engineering.
    """
    print("="*70)
    print("PURCHASING POWER PRESERVATION - CONSOLIDATED DATA GENERATION")
    print("="*70)

    # 1. Fetch Raw Data
    df_economic = fetch_economic_data()
    df_assets_vix = fetch_asset_and_vix_prices()
    df_crypto_prices = fetch_crypto_data_yfinance()
    df_crypto_supply = fetch_crypto_supply_yfinance()
    df_stock_supply = fetch_stock_supply_data() # Not directly merged, but used for market cap config
    df_commodities_fred = fetch_real_baselines()
    df_global_m2, df_global_gdp = fetch_global_market_data()

    # 2. Merge Raw Data
    df_merged = merge_all_raw_data(df_economic, df_assets_vix, df_crypto_prices, df_commodities_fred, df_global_m2, df_global_gdp)
    
    # Prepare asset configurations for comprehensive feature engineering
    crypto_supplies_dict = {row['Asset']: {'circulating': row['Circulating_Supply'], 'max': row['Max_Supply']}
                            for idx, row in df_crypto_supply.iterrows()}
    
    asset_configs = get_all_asset_configs(df_merged, crypto_supplies_dict)

    # 3. Comprehensive Feature Engineering
    df_features = calculate_all_returns_volatility_technicals(df_merged.copy(), asset_configs)
    df_features = engineer_core_features(df_features)
    df_features = add_purchasing_power_multipliers_and_baselines(df_features, asset_configs)
    df_features = add_economic_rules(df_features)
    df_features = add_market_cap_saturation_and_risk(df_features, df_global_m2, df_global_gdp, df_crypto_supply)

    # 4. Create Labels
    df_final = create_labels(df_features, asset_configs)

    # 5. Round Numerical Columns
    df_final_rounded = round_all_numerical_columns(df_final.copy(), decimals=4)

    # Save final dataset
    final_output_path = f"{OUTPUT_DIR}/raw/final_consolidated_dataset.csv"
    df_final_rounded.to_csv(final_output_path, index=False)
    print(f"\n✅ Final consolidated dataset saved to: {final_output_path}")

    # Summary statistics
    print("\n" + "="*70)
    print("DATA GENERATION SUMMARY")
    print("="*70)
    print(f"★ Date Range: {df_final_rounded['Date'].min()} to {df_final_rounded['Date'].max()}")
    print(f"★ Total Rows: {len(df_final_rounded):,}")
    print(f"★ Total Columns: {len(df_final_rounded.columns)}")
    print(f"★ Total Size: {df_final_rounded.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print(f"\n📁 Files saved in: {OUTPUT_DIR}")
    print("\n✅ Consolidated data generation complete!")
    print("="*70)

    return df_final_rounded

# ========================================
# RUN THE SCRIPT
# ========================================

if __name__ == "__main__":
    df_final_dataset = generate_consolidated_dataset()

    # Display sample
    print("\n📋 Sample of final consolidated dataset (head 5 rows):")
    print(df_final_dataset.head())

    # Display last 5 rows
    print("\n📋 Sample of final consolidated dataset (tail 5 rows):") # Corrected syntax error by removing the extra backtick
    print(df_final_dataset.tail())

    # Check for missing values
    print("\n❓ Missing values (top 20):")
    missing = df_final_dataset.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(missing.head(20))
    else:
        print("  No missing values!")

    # Display data types
    print("\n🔢 Data types (value counts):")
    print(df_final_dataset.dtypes.value_counts())