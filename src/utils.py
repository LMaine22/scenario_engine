"""
Utilities: Data loading, preprocessing, and helpers
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_yield_data(config):
    """Load and prepare yield curve data"""
    df = pd.read_parquet(config['data']['raw_path'])
    
    # Handle date index - if it's in 'Date' column, set it as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], unit='ns' if df['Date'].dtype == 'int64' else None)
        df = df.set_index('Date')
    else:
        df.index = pd.to_datetime(df.index, unit='ns' if df.index.dtype == 'int64' else None)
    
    df = df.sort_index()
    
    # Extract only the tenors we need
    tenors = config['data']['tenors']
    df = df[tenors + ['MOVE Index', 'VIX Index', 'FEDL01 Index',
                      'USOSFR1 Curncy', 'USOSFR2 Curncy', 'USOSFR5 Curncy']]
    
    # Drop any rows with missing values
    df = df.dropna()
    
    return df


def load_econ_data(config):
    """Load and prepare economic surprise data"""
    reports = pd.read_parquet(config['data']['econ_reports'])
    jobs = pd.read_parquet(config['data']['econ_jobs'])
    
    # Combine both datasets
    econ = pd.concat([reports, jobs], ignore_index=True)
    econ['Date'] = pd.to_datetime(econ['Date'], format='mixed', errors='coerce')
    econ = econ.dropna(subset=['Date'])  # Drop any rows with invalid dates
    econ = econ.sort_values('Date')
    
    return econ


def calculate_surprises(econ_df):
    """Calculate standardized economic surprises"""
    # Convert Survey and Actual to numeric
    econ_df['Survey_num'] = pd.to_numeric(econ_df['Survey'], errors='coerce')
    econ_df['Actual_num'] = pd.to_numeric(econ_df['Actual'], errors='coerce')
    
    # Calculate raw surprise
    econ_df['Surprise'] = econ_df['Actual_num'] - econ_df['Survey_num']
    
    # Standardize by event type
    econ_df['Surprise_z'] = econ_df.groupby('Event')['Surprise'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    return econ_df


def build_regime_features(df, config):
    """Build regime embedding features"""
    window = config['regime']['vol_window']
    
    # 1. MOVE z-score
    df['MOVE_z'] = (df['MOVE Index'] - df['MOVE Index'].rolling(window).mean()) / \
                    df['MOVE Index'].rolling(window).std()
    
    # 2. VIX z-score
    df['VIX_z'] = (df['VIX Index'] - df['VIX Index'].rolling(window).mean()) / \
                   df['VIX Index'].rolling(window).std()
    
    # 3. OIS slope (5yr - 1yr)
    df['OIS_slope'] = df['USOSFR5 Curncy'] - df['USOSFR1 Curncy']
    
    # 4. Fed vs OIS gap (actual - expected)
    df['Fed_vs_OIS'] = df['FEDL01 Index'] - df['USOSFR1 Curncy']
    
    # 5. Inflation surprise memory (will add after merging econ data)
    
    return df


def merge_econ_surprises(yield_df, econ_df, config):
    """Merge economic surprises with yield data"""
    # Calculate surprises
    econ_df = calculate_surprises(econ_df)
    
    # Filter for inflation-related events
    inflation_events = ['CPI', 'Core CPI', 'PPI', 'Core PPI', 'PCE']
    inflation_df = econ_df[econ_df['Event'].str.contains('|'.join(inflation_events), na=False)]
    
    # Aggregate surprises by date
    daily_surprise = inflation_df.groupby(inflation_df['Date'].dt.date).agg({
        'Surprise_z': 'mean',
        'Relevance': 'sum'
    }).reset_index()
    daily_surprise['Date'] = pd.to_datetime(daily_surprise['Date'])
    
    # Merge with yield data
    yield_df = yield_df.merge(daily_surprise, left_index=True, right_on='Date', how='left')
    yield_df['Surprise_z'] = yield_df['Surprise_z'].fillna(0)
    
    # Calculate EWMA inflation memory
    halflife = config['regime']['inflation_halflife']
    yield_df['Inflation_mem'] = yield_df['Surprise_z'].ewm(halflife=halflife).mean()
    
    yield_df = yield_df.drop(columns=['Date', 'Relevance'], errors='ignore')
    
    return yield_df


def prepare_data(config):
    """Main data preparation pipeline"""
    # Load data
    yield_df = load_yield_data(config)
    econ_df = load_econ_data(config)
    
    # Build regime features
    yield_df = build_regime_features(yield_df, config)
    
    # Merge economic surprises
    yield_df = merge_econ_surprises(yield_df, econ_df, config)
    
    # Drop rows with NaN in regime features (from rolling windows)
    regime_cols = ['MOVE_z', 'VIX_z', 'OIS_slope', 'Fed_vs_OIS', 'Inflation_mem']
    yield_df = yield_df.dropna(subset=regime_cols)
    
    # Save processed data
    output_path = Path(config['data']['processed_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    yield_df.to_parquet(output_path / "processed_data.parquet")
    
    print(f"Processed data: {len(yield_df)} observations")
    print(f"Date range: {yield_df.index.min()} to {yield_df.index.max()}")
    
    return yield_df


def get_yield_matrix(df, config):
    """Extract yield matrix (dates x tenors)"""
    tenors = config['data']['tenors']
    return df[tenors].values


def get_regime_matrix(df):
    """Extract regime features matrix"""
    regime_cols = ['MOVE_z', 'VIX_z', 'OIS_slope', 'Fed_vs_OIS', 'Inflation_mem']
    return df[regime_cols].values
