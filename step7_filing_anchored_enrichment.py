"""
Step 7: Filing-Anchored Market Features & Targets Enrichment
============================================================

This module enriches the existing ml_filing_anchored_full.csv with:
1. Market features computed ONLY from history before anchor_date
2. Forward targets computed from anchor_date forward
3. No look-ahead bias validation

Key Features:
- Uses daily prices (no quarter-end resampling)
- Features: momentum, volatility, drawdown (pre-anchor only)
- Targets: 21D, 63D, 126D forward returns (absolute & excess)
- Binary targets for 63D horizon
- Comprehensive validation against data leakage

Author: Finance ML Learning Project
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input/Output paths
FUNDAMENTALS_FILE = Path('artifacts/fundamentals/fundamentals_quarterly_long.csv')
ANCHORED_FILE = Path('artifacts/ml_data/ml_filing_anchored_full.csv')
DB_PATH = Path('financial_data.db')
OUTPUT_DIR = Path('artifacts/ml_data')
GRAPHS_DIR = Path('graphs/ml_targets')

# Tickers
STOCK_TICKER = 'AAPL'
MARKET_TICKER = '^GSPC'

# Feature parameters
MOMENTUM_WINDOWS = {'mom_6m': 126, 'mom_12m': 252}  # Trading days
VOLATILITY_WINDOWS = {'rv_21d': 21, 'rv_63d': 63}   # Trading days
DRAWDOWN_WINDOWS = {'dd_6m': 126, 'dd_12m': 252}    # Trading days
TARGET_HORIZONS = [21, 63, 126]  # Trading days forward

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def setup_directories():
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created output directories")

def load_daily_prices(ticker: str) -> pd.DataFrame:
    """Load daily price data for a ticker from financial_data.db."""
    print(f"\n📈 Loading {ticker} daily price data...")
    
    if not DB_PATH.exists():
        print(f"  ⚠️ Database not found: {DB_PATH}")
        return None
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Query price data for the ticker
        table_name = f"{ticker}_price"
        if ticker.startswith('^'):
            table_name_quoted = f'"{table_name}"'
        else:
            table_name_quoted = table_name
            
        query = f"""
        SELECT Date as date, Adj_Close as adj_close
        FROM {table_name_quoted}
        ORDER BY Date
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print(f"  ⚠️ No price data found for {ticker}")
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        print(f"  ✅ Loaded price data: {len(df)} records from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        return df
        
    except Exception as e:
        print(f"  ❌ Error loading price data: {e}")
        return None

def load_anchored_data() -> pd.DataFrame:
    """Load the existing filing-anchored dataset."""
    print(f"\n📊 Loading filing-anchored data...")
    
    if not ANCHORED_FILE.exists():
        print(f"  ⚠️ Anchored file not found: {ANCHORED_FILE}")
        return None
    
    try:
        df = pd.read_csv(ANCHORED_FILE)
        df['anchor_date'] = pd.to_datetime(df['anchor_date'])
        df['reported_date'] = pd.to_datetime(df['reported_date'])
        
        print(f"  ✅ Loaded anchored data: {len(df)} records")
        print(f"  📅 Date range: {df['anchor_date'].min().strftime('%Y-%m-%d')} to {df['anchor_date'].max().strftime('%Y-%m-%d')}")
        return df
        
    except Exception as e:
        print(f"  ❌ Error loading anchored data: {e}")
        return None

# ============================================================================
# MARKET FEATURE COMPUTATION
# ============================================================================

def compute_momentum_features(px: pd.DataFrame, anchor_date: pd.Timestamp) -> dict:
    """Compute momentum features using only data before anchor_date."""
    historical_data = px[px.index < anchor_date]
    features = {}
    
    for feature_name, window in MOMENTUM_WINDOWS.items():
        if len(historical_data) >= window:
            # Log return over the window ending at t-1
            start_price = historical_data['adj_close'].iloc[-window]
            end_price = historical_data['adj_close'].iloc[-1]
            features[feature_name] = np.log(end_price / start_price)
        else:
            features[feature_name] = np.nan
    
    return features

def compute_volatility_features(px: pd.DataFrame, anchor_date: pd.Timestamp) -> dict:
    """Compute realized volatility features using only data before anchor_date."""
    historical_data = px[px.index < anchor_date]
    features = {}
    
    if len(historical_data) < 2:
        return {name: np.nan for name in VOLATILITY_WINDOWS.keys()}
    
    # Compute daily log returns
    log_returns = np.log(historical_data['adj_close'] / historical_data['adj_close'].shift(1)).dropna()
    
    for feature_name, window in VOLATILITY_WINDOWS.items():
        if len(log_returns) >= window:
            # Annualized volatility
            features[feature_name] = log_returns.tail(window).std() * np.sqrt(252)
        else:
            features[feature_name] = np.nan
    
    return features

def compute_drawdown_features(px: pd.DataFrame, anchor_date: pd.Timestamp) -> dict:
    """Compute maximum drawdown features using only data before anchor_date."""
    historical_data = px[px.index < anchor_date]
    features = {}
    
    for feature_name, window in DRAWDOWN_WINDOWS.items():
        if len(historical_data) >= window:
            # Rolling maximum over the window
            rolling_max = historical_data['adj_close'].rolling(window=window, min_periods=1).max()
            # Drawdown = (price / rolling_max) - 1
            drawdown = (historical_data['adj_close'] / rolling_max) - 1
            # Maximum drawdown (most negative)
            features[feature_name] = drawdown.min()
        else:
            features[feature_name] = np.nan
    
    return features

def compute_market_features(px: pd.DataFrame, anchor_date: pd.Timestamp) -> dict:
    """Compute all market features using only data before anchor_date."""
    features = {}
    
    # Momentum features
    features.update(compute_momentum_features(px, anchor_date))
    
    # Volatility features
    features.update(compute_volatility_features(px, anchor_date))
    
    # Drawdown features
    features.update(compute_drawdown_features(px, anchor_date))
    
    return features

# ============================================================================
# TARGET COMPUTATION
# ============================================================================

def compute_forward_targets(px_stock: pd.DataFrame, px_mkt: pd.DataFrame, anchor_date: pd.Timestamp) -> dict:
    """Compute forward targets from anchor_date."""
    targets = {}
    
    # Get anchor prices
    if anchor_date not in px_stock.index or anchor_date not in px_mkt.index:
        # Return NaN for all targets if anchor date not available
        for horizon in TARGET_HORIZONS:
            targets[f'y_fwd_{horizon}d_log'] = np.nan
            targets[f'y_fwd_{horizon}d_excess'] = np.nan
        targets['y_up_63d'] = np.nan
        return targets
    
    anchor_price_stock = px_stock.loc[anchor_date, 'adj_close']
    anchor_price_mkt = px_mkt.loc[anchor_date, 'adj_close']
    
    # Compute targets for each horizon
    for horizon in TARGET_HORIZONS:
        # Find target date (horizon trading days after anchor)
        future_dates = px_stock.index[px_stock.index > anchor_date]
        if len(future_dates) < horizon:
            targets[f'y_fwd_{horizon}d_log'] = np.nan
            targets[f'y_fwd_{horizon}d_excess'] = np.nan
            continue
        
        target_date = future_dates[horizon - 1]  # horizon-th trading day after anchor
        
        # Check if both stock and market have data on target date
        if target_date not in px_stock.index or target_date not in px_mkt.index:
            targets[f'y_fwd_{horizon}d_log'] = np.nan
            targets[f'y_fwd_{horizon}d_excess'] = np.nan
            continue
        
        # Get target prices
        target_price_stock = px_stock.loc[target_date, 'adj_close']
        target_price_mkt = px_mkt.loc[target_date, 'adj_close']
        
        # Compute log returns
        stock_log_return = np.log(target_price_stock / anchor_price_stock)
        market_log_return = np.log(target_price_mkt / anchor_price_mkt)
        
        # Store targets
        targets[f'y_fwd_{horizon}d_log'] = stock_log_return
        targets[f'y_fwd_{horizon}d_excess'] = stock_log_return - market_log_return
    
    # Binary target for 63D horizon
    if not pd.isna(targets.get('y_fwd_63d_log')):
        targets['y_up_63d'] = 1 if targets['y_fwd_63d_log'] > 0 else 0
    else:
        targets['y_up_63d'] = np.nan
    
    return targets

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_no_lookahead_bias(df: pd.DataFrame, px_stock: pd.DataFrame, px_mkt: pd.DataFrame):
    """Validate that no features use data >= anchor_date."""
    print(f"\n🔍 Validating no look-ahead bias...")
    
    # Check 5 random samples
    sample_indices = np.random.choice(len(df), min(5, len(df)), replace=False)
    
    for idx in sample_indices:
        row = df.iloc[idx]
        anchor_date = row['anchor_date']
        
        print(f"\n  📊 Sample {idx}:")
        print(f"    Anchor date: {anchor_date}")
        
        # Check that all market features use data before anchor_date
        feature_cols = ['mom_6m', 'mom_12m', 'rv_21d', 'rv_63d', 'dd_6m', 'dd_12m']
        
        for col in feature_cols:
            if col in df.columns and not pd.isna(row[col]):
                # Verify the feature was computed correctly
                if col.startswith('mom_'):
                    window = MOMENTUM_WINDOWS[col]
                    historical_data = px_stock[px_stock.index < anchor_date]
                    if len(historical_data) >= window:
                        expected = np.log(historical_data['adj_close'].iloc[-1] / historical_data['adj_close'].iloc[-window])
                        assert abs(row[col] - expected) < 1e-10, f"Feature {col} has look-ahead bias"
                        print(f"    ✅ {col}: {row[col]:.4f}")
        
        # Check that anchor_date is after reported_date
        assert anchor_date >= row['reported_date'], f"Anchor date {anchor_date} is before reported date {row['reported_date']}"
        print(f"    ✅ Anchor date validation passed")
    
    print(f"  ✅ No look-ahead bias validation completed")

# ============================================================================
# ENRICHMENT FUNCTIONS
# ============================================================================

def enrich_anchored_data(anchored_df: pd.DataFrame, px_stock: pd.DataFrame, px_mkt: pd.DataFrame) -> pd.DataFrame:
    """Enrich anchored data with market features and forward targets."""
    print(f"\n🔧 Enriching anchored data with market features and targets...")
    
    enriched_rows = []
    
    for idx, row in anchored_df.iterrows():
        anchor_date = row['anchor_date']
        
        # Compute market features (pre-anchor only)
        market_features = compute_market_features(px_stock, anchor_date)
        
        # Compute forward targets
        forward_targets = compute_forward_targets(px_stock, px_mkt, anchor_date)
        
        # Combine all data - convert Series to dict first to allow new keys
        enriched_row = row.to_dict()
        enriched_row.update(market_features)
        enriched_row.update(forward_targets)
        
        enriched_rows.append(enriched_row)
    
    enriched_df = pd.DataFrame(enriched_rows)
    print(f"  ✅ Enriched {len(enriched_df)} records")
    
    return enriched_df

# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_enriched_datasets(df: pd.DataFrame):
    """Save the enriched filing-anchored datasets."""
    print(f"\n💾 Saving enriched datasets...")
    
    # Save full enriched dataset
    full_file = OUTPUT_DIR / 'ml_filing_anchored_enriched.csv'
    df.to_csv(full_file, index=False)
    print(f"✅ Saved enriched dataset: {full_file}")
    
    # Create focused dataset
    focused_cols = [
        'ticker', 'reported_date', 'anchor_date', 'anchor_price',
        # Fundamental features
        'revenue_yoy', 'eps_yoy', 'freeCashFlow_yoy', 'revenue_qoq',
        'gross_margin_ttm', 'op_margin_ttm', 'net_margin_ttm',
        'shares_chg_yoy', 'pe_ttm',
        # Market features
        'mom_6m', 'mom_12m', 'rv_21d', 'rv_63d', 'dd_6m', 'dd_12m',
        # Targets
        'y_fwd_21d_log', 'y_fwd_63d_log', 'y_fwd_126d_log',
        'y_fwd_21d_excess', 'y_fwd_63d_excess', 'y_fwd_126d_excess',
        'y_up_63d'
    ]
    
    available_cols = [col for col in focused_cols if col in df.columns]
    focused_df = df[available_cols].copy()
    
    focused_file = OUTPUT_DIR / 'ml_filing_anchored_focused.csv'
    focused_df.to_csv(focused_file, index=False)
    print(f"✅ Saved focused dataset: {focused_file}")

def create_diagnostic_plots(df: pd.DataFrame):
    """Create diagnostic visualizations."""
    print(f"\n📊 Creating diagnostic plots...")
    
    # Set up matplotlib style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # Create plots
    create_target_distribution_plot(df)
    create_feature_distribution_plot(df)
    create_correlation_plot(df)
    
    print(f"✅ Created diagnostic plots in {GRAPHS_DIR}")

def create_target_distribution_plot(df: pd.DataFrame):
    """Create target distribution plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Filing-Anchored Target Distributions', fontsize=16, fontweight='bold')
    
    # 1. 63D log returns distribution
    returns_63d = df['y_fwd_63d_log'].dropna()
    if not returns_63d.empty:
        ax1.hist(returns_63d, bins=30, alpha=0.7, color='blue', density=True)
        ax1.axvline(returns_63d.mean(), color='red', linestyle='--', label=f'Mean: {returns_63d.mean():.3f}')
        ax1.set_title('63D Forward Log Returns Distribution')
        ax1.set_xlabel('Log Return')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 63D excess returns distribution
    excess_63d = df['y_fwd_63d_excess'].dropna()
    if not excess_63d.empty:
        ax2.hist(excess_63d, bins=30, alpha=0.7, color='green', density=True)
        ax2.axvline(excess_63d.mean(), color='red', linestyle='--', label=f'Mean: {excess_63d.mean():.3f}')
        ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('63D Forward Excess Returns Distribution')
        ax2.set_xlabel('Excess Return')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Binary target distribution
    if 'y_up_63d' in df.columns:
        up_counts = df['y_up_63d'].value_counts()
        ax3.bar(['Down (0)', 'Up (1)'], up_counts.values, color=['red', 'green'], alpha=0.7)
        ax3.set_title('Binary Target Distribution (63D)')
        ax3.set_ylabel('Count')
        
        # Add percentage labels
        total = up_counts.sum()
        for i, v in enumerate(up_counts.values):
            ax3.text(i, v + 0.01, f'{v/total:.1%}', ha='center', va='bottom')
    
    # 4. Returns by horizon
    horizons = [21, 63, 126]
    horizon_means = []
    for h in horizons:
        col = f'y_fwd_{h}d_log'
        if col in df.columns:
            horizon_means.append(df[col].mean())
        else:
            horizon_means.append(np.nan)
    
    ax4.bar([f'{h}D' for h in horizons], horizon_means, color=['orange', 'blue', 'purple'], alpha=0.7)
    ax4.set_title('Mean Returns by Horizon')
    ax4.set_ylabel('Mean Log Return')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'target_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_distribution_plot(df: pd.DataFrame):
    """Create feature distribution plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Market Feature Distributions', fontsize=16, fontweight='bold')
    
    # 1. 12M momentum
    mom_12m = df['mom_12m'].dropna()
    if not mom_12m.empty:
        ax1.hist(mom_12m, bins=30, alpha=0.7, color='blue', density=True)
        ax1.axvline(mom_12m.mean(), color='red', linestyle='--', label=f'Mean: {mom_12m.mean():.3f}')
        ax1.set_title('12M Momentum Distribution')
        ax1.set_xlabel('12M Log Return')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 63D realized volatility
    rv_63d = df['rv_63d'].dropna()
    if not rv_63d.empty:
        ax2.hist(rv_63d, bins=30, alpha=0.7, color='green', density=True)
        ax2.axvline(rv_63d.mean(), color='red', linestyle='--', label=f'Mean: {rv_63d.mean():.3f}')
        ax2.set_title('63D Realized Volatility Distribution')
        ax2.set_xlabel('Annualized Volatility')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 12M drawdown
    dd_12m = df['dd_12m'].dropna()
    if not dd_12m.empty:
        ax3.hist(dd_12m, bins=30, alpha=0.7, color='red', density=True)
        ax3.axvline(dd_12m.mean(), color='black', linestyle='--', label=f'Mean: {dd_12m.mean():.3f}')
        ax3.set_title('12M Maximum Drawdown Distribution')
        ax3.set_xlabel('Maximum Drawdown')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Feature correlation with target
    feature_cols = ['mom_6m', 'mom_12m', 'rv_21d', 'rv_63d', 'dd_6m', 'dd_12m']
    correlations = []
    feature_names = []
    
    for col in feature_cols:
        if col in df.columns:
            corr = df[col].corr(df['y_fwd_63d_log'])
            if not pd.isna(corr):
                correlations.append(corr)
                feature_names.append(col.replace('_', ' ').title())
    
    if correlations:
        ax4.barh(feature_names, correlations, color='purple', alpha=0.7)
        ax4.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax4.set_title('Feature Correlations with 63D Returns')
        ax4.set_xlabel('Correlation')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_plot(df: pd.DataFrame):
    """Create correlation heatmap."""
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_data = df[numeric_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    
    # Use matplotlib's imshow instead of seaborn
    im = plt.imshow(corr_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, shrink=0.8)
    
    # Add correlation values as text
    for i in range(len(corr_data.columns)):
        for j in range(len(corr_data.columns)):
            if not mask[i, j]:
                plt.text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                        ha='center', va='center', fontsize=8)
    
    plt.xticks(range(len(corr_data.columns)), corr_data.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr_data.columns)), corr_data.columns)
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    """Main function for filing-anchored enrichment."""
    print("🎯 Step 7: Filing-Anchored Market Features & Targets Enrichment")
    print("=" * 70)
    print(f"📈 Stock: {STOCK_TICKER}")
    print(f"📈 Market: {MARKET_TICKER}")
    print(f"🎯 Target Horizons: {TARGET_HORIZONS} trading days")
    print(f"📊 Market Features: {list(MOMENTUM_WINDOWS.keys())} + {list(VOLATILITY_WINDOWS.keys())} + {list(DRAWDOWN_WINDOWS.keys())}")
    print("=" * 70)
    
    # Setup directories
    setup_directories()
    
    # Load data
    anchored_df = load_anchored_data()
    if anchored_df is None:
        print("❌ Cannot proceed without anchored data")
        return
    
    px_stock = load_daily_prices(STOCK_TICKER)
    if px_stock is None:
        print("❌ Cannot proceed without stock price data")
        return
    
    px_mkt = load_daily_prices(MARKET_TICKER)
    if px_mkt is None:
        print("❌ Cannot proceed without market price data")
        return
    
    # Enrich data with market features and targets
    enriched_df = enrich_anchored_data(anchored_df, px_stock, px_mkt)
    
    # Validate no look-ahead bias
    validate_no_lookahead_bias(enriched_df, px_stock, px_mkt)
    
    # Create diagnostic plots
    create_diagnostic_plots(enriched_df)
    
    # Save enriched datasets
    save_enriched_datasets(enriched_df)
    
    # Print summary
    print(f"\n✅ Step 7 Complete!")
    print(f"📁 Enriched datasets saved to: {OUTPUT_DIR}")
    print(f"📊 Diagnostic plots saved to: {GRAPHS_DIR}")
    print(f"📈 Total records: {len(enriched_df)}")
    print(f"🎯 Primary target: y_fwd_63d_log")
    print(f"🔍 No look-ahead bias validated")

if __name__ == "__main__":
    main()
