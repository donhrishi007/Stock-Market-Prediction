"""
Stock Analysis Module - STEP 2 (FMP Data Optimized)
==================================================

📊 EXECUTION ORDER: RUN THIS FILE SECOND (after FMP data download)!

This module handles comprehensive analysis of FMP stock data including:
- Apple vs S&P 500 comparison analysis (35+ years of data)
- MAG7 stocks analysis and comparison (complete historical coverage)
- Annual returns calculations with extended historical analysis
- Statistical analysis and visualizations optimized for FMP data
- Correlation analysis across multiple market cycles
- Enhanced performance metrics with professional-grade data

Features:
- Optimized for 35+ years of historical data (1990-2025)
- Enhanced analysis with FMP professional-grade data
- Comprehensive market cycle analysis
- Advanced statistical metrics and visualizations

Author: Finance ML Learning Project
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sqlalchemy import create_engine, inspect
import os
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define the MAG7 stocks
MAG7_TICKERS = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META']

# Database configuration
DB_NAME = 'financial_data.db'

# Create graphs directory
GRAPHS_DIR = os.path.join(os.path.dirname(__file__), 'graphs', 'stock_analysis')
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def connect_to_database():
    """Connect to the SQLite database containing FMP stock data."""
    print("🔌 Connecting to FMP database...")
    
    db_path = os.path.join(os.path.dirname(__file__), DB_NAME)
    engine = create_engine(f'sqlite:///{db_path}')
    inspector = inspect(engine)
    
    # Get table information
    tables = inspector.get_table_names()
    print(f"✅ Connected to database: {DB_NAME}")
    print(f"📊 Found {len(tables)} tables with FMP data")
    
    return engine, inspector

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_stock_data(engine, inspector, ticker):
    """Load FMP stock data from database for a specific ticker."""
    print(f"📊 Loading {ticker} FMP price data...")
    
    table_name = f'{ticker}_price'
    
    if table_name in inspector.get_table_names():
        df = pd.read_sql_table(table_name, engine)
        
        # Clean up column names
        df.columns = ['_'.join(col) if isinstance(col, tuple) else str(col) 
                     for col in df.columns]
        
        # Convert Date column to datetime and set as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.set_index('Date')
            df = df[~df.index.isnull()]
        
        # Calculate years of data
        years_of_data = (df.index.max() - df.index.min()).days / 365.25
        
        print(f"✅ Loaded {ticker} FMP data: {df.shape[0]:,} rows")
        print(f"📅 Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        print(f"📈 Years of data: {years_of_data:.1f} years")
        print(f"💰 Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        return df
    else:
        print(f"❌ Table {table_name} not found in FMP database")
        return None

def load_all_mag7_data(engine, inspector):
    """Load data for all MAG7 stocks."""
    print("\n📊 Loading MAG7 stocks data...")
    
    mag7_data = {}
    for ticker in MAG7_TICKERS:
        data = load_stock_data(engine, inspector, ticker)
        if data is not None:
            mag7_data[ticker] = data
    
    return mag7_data

# ============================================================================
# ANNUAL RETURNS CALCULATION
# ============================================================================

def calculate_annual_returns(df, ticker):
    """Calculate annual returns for a stock."""
    print(f"📈 Calculating {ticker} annual returns...")
    
    # Find the Close column (it might be named Close_TICKER)
    close_col = [col for col in df.columns if 'Close' in col][0]
    annual_prices = df[close_col].resample('Y').last()
    
    # Calculate annual returns
    annual_returns = annual_prices.pct_change() * 100
    
    # Create DataFrame with Year and Annual_Return columns
    annual_data = pd.DataFrame({
        'Year': annual_returns.index.year,
        'Annual_Return': annual_returns.values
    })
    
    # Remove NaN values (first year will have NaN)
    annual_data = annual_data.dropna()
    
    print(f"📊 Calculated {len(annual_data)} years of {ticker} annual returns")
    print(f"📅 Year range: {annual_data['Year'].min()} to {annual_data['Year'].max()}")
    
    return annual_data

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def calculate_statistics(annual_data, ticker):
    """Calculate comprehensive statistics for annual returns."""
    print(f"📊 Calculating {ticker} statistics...")
    
    returns = annual_data['Annual_Return']
    
    # Basic statistics
    mean_return = returns.mean()
    std_return = returns.std()
    median_return = returns.median()
    
    # Percentiles
    percentiles = returns.quantile([0.05, 0.25, 0.75, 0.95])
    
    # Confidence intervals (68% = ±1 std dev)
    confidence_68_lower = mean_return - std_return
    confidence_68_upper = mean_return + std_return
    
    # Skewness and kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # Best and worst years
    best_year = annual_data.loc[returns.idxmax()]
    worst_year = annual_data.loc[returns.idxmin()]
    
    stats_dict = {
        'mean': mean_return,
        'std': std_return,
        'median': median_return,
        'percentiles': percentiles,
        'confidence_68_lower': confidence_68_lower,
        'confidence_68_upper': confidence_68_upper,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'best_year': best_year,
        'worst_year': worst_year,
        'total_years': len(returns)
    }
    
    print(f"📈 {ticker} Mean Annual Return: {mean_return:.2f}%")
    print(f"📊 {ticker} Annualized Volatility (Std Dev): {std_return:.2f}%")
    print(f"📉 {ticker} Median Return: {median_return:.2f}%")
    
    return stats_dict

# ============================================================================
# APPLE VS S&P 500 ANALYSIS
# ============================================================================

def create_apple_vs_sp500_analysis():
    """Create comprehensive Apple vs S&P 500 analysis."""
    print("🍎 Creating Apple vs S&P 500 analysis...")
    
    # Connect to database
    engine, inspector = connect_to_database()
    
    # Load Apple data
    aapl_data = load_stock_data(engine, inspector, 'AAPL')
    if aapl_data is None:
        print("❌ Could not load Apple data")
        return
    
    # Load S&P 500 data (try different possible tickers)
    sp500_tickers = ['^GSPC', 'GSPC', 'SP500', 'SPX']
    sp500_data = None
    sp500_ticker = None
    
    for ticker in sp500_tickers:
        data = load_stock_data(engine, inspector, ticker)
        if data is not None:
            sp500_data = data
            sp500_ticker = ticker
            print(f"✅ Found S&P 500 data with ticker: {ticker}")
            break
    
    if sp500_data is None:
        print("❌ Could not load S&P 500 data")
        return
    
    # Calculate annual returns
    aapl_annual = calculate_annual_returns(aapl_data, 'AAPL')
    sp500_annual = calculate_annual_returns(sp500_data, sp500_ticker)
    
    # Calculate statistics
    aapl_stats = calculate_statistics(aapl_annual, 'AAPL')
    sp500_stats = calculate_statistics(sp500_annual, sp500_ticker)
    
    # Create visualizations
    create_comparison_visualization(aapl_annual, sp500_annual, aapl_stats, sp500_stats)
    create_focused_comparison_plot(aapl_annual, sp500_annual, aapl_stats, sp500_stats)
    create_standardized_returns_analysis(aapl_annual, sp500_annual, aapl_stats, sp500_stats)
    
    # Print analysis
    print_comparison_analysis(aapl_annual, sp500_annual, aapl_stats, sp500_stats)
    
    return aapl_annual, sp500_annual, aapl_stats, sp500_stats

# ============================================================================
# MAG7 ANALYSIS
# ============================================================================

def create_mag7_analysis():
    """Create comprehensive MAG7 stocks analysis."""
    print("🚀 Creating MAG7 stocks analysis...")
    
    # Connect to database
    engine, inspector = connect_to_database()
    
    # Load all MAG7 data
    mag7_data = load_all_mag7_data(engine, inspector)
    
    if not mag7_data:
        print("❌ No MAG7 data loaded")
        return
    
    # Calculate annual returns for all stocks
    mag7_annual_data = {}
    mag7_stats = {}
    
    for ticker in MAG7_TICKERS:
        if ticker in mag7_data:
            annual_data = calculate_annual_returns(mag7_data[ticker], ticker)
            stats = calculate_statistics(annual_data, ticker)
            
            mag7_annual_data[ticker] = annual_data
            mag7_stats[ticker] = stats
    
    # Create Apple vs each MAG7 stock detailed comparisons
    if 'AAPL' in mag7_annual_data:
        aapl_data = mag7_annual_data['AAPL']
        aapl_stats = mag7_stats['AAPL']
        
        for ticker in MAG7_TICKERS:
            if ticker != 'AAPL' and ticker in mag7_annual_data:
                print(f"\n🍎 Creating Apple vs {ticker} detailed comparison...")
                create_apple_vs_stock_comparison(aapl_data, mag7_annual_data[ticker], 
                                               aapl_stats, mag7_stats[ticker], ticker)
    
    # Create MAG7 comparison visualization
    if len(mag7_annual_data) > 1:
        create_mag7_comparison_visualization(mag7_annual_data, mag7_stats)
    
    return mag7_annual_data, mag7_stats

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comparison_visualization(aapl_data, sp500_data, aapl_stats, sp500_stats):
    """Create the main comparison visualization."""
    print("📊 Creating comparison visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    returns_aapl = aapl_data['Annual_Return']
    returns_sp500 = sp500_data['Annual_Return']
    
    # Plot 1: Annual returns comparison
    x_min, x_max = -60, 120
    bins_aapl = np.linspace(x_min, x_max, 20)
    bins_sp500 = np.linspace(x_min, x_max, 15)
    
    ax1.hist(returns_aapl, bins=bins_aapl, alpha=0.7, edgecolor='black', 
             color='lightblue', density=True, label='Apple (AAPL) Returns')
    ax1.hist(returns_sp500, bins=bins_sp500, alpha=0.7, edgecolor='black', 
             color='lightcoral', density=True, label='S&P 500 (^GSPC) Returns')
    
    # Add normal distribution overlays
    x_range = np.linspace(x_min, x_max, 200)
    normal_aapl = stats.norm.pdf(x_range, aapl_stats['mean'], aapl_stats['std'])
    normal_sp500 = stats.norm.pdf(x_range, sp500_stats['mean'], sp500_stats['std'])
    
    ax1.plot(x_range, normal_aapl, 'b-', linewidth=3, label='AAPL Normal Distribution')
    ax1.plot(x_range, normal_sp500, 'r-', linewidth=3, label='S&P 500 Normal Distribution')
    
    # Add mean lines
    ax1.axvline(aapl_stats['mean'], color='blue', linestyle='--', linewidth=2, 
                label=f'AAPL Mean: {aapl_stats["mean"]:.1f}%')
    ax1.axvline(sp500_stats['mean'], color='red', linestyle='--', linewidth=2, 
                label=f'S&P 500 Mean: {sp500_stats["mean"]:.1f}%')
    
    ax1.set_title('Returns Distribution Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Annual Return (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Returns over time
    ax2.plot(aapl_data['Year'], returns_aapl, 'o-', linewidth=2, markersize=6, 
             label='Apple (AAPL)', color='blue')
    ax2.plot(sp500_data['Year'], returns_sp500, 's-', linewidth=2, markersize=6, 
             label='S&P 500 (^GSPC)', color='red')
    
    ax2.axhline(y=aapl_stats['mean'], color='blue', linestyle='--', alpha=0.7, 
                label=f'AAPL Mean: {aapl_stats["mean"]:.1f}%')
    ax2.axhline(y=sp500_stats['mean'], color='red', linestyle='--', alpha=0.7, 
                label=f'S&P 500 Mean: {sp500_stats["mean"]:.1f}%')
    
    ax2.set_title('Annual Returns Over Time', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Annual Return (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Rolling Correlation
    combined_data = pd.merge(aapl_data, sp500_data, on='Year', suffixes=('_AAPL', '_SP500'))
    if len(combined_data) >= 5:
        rolling_corr = combined_data['Annual_Return_AAPL'].rolling(window=5, min_periods=1).corr(combined_data['Annual_Return_SP500'])
        
        ax3.plot(combined_data['Year'], rolling_corr, 'g-', linewidth=2, label='5-Year Rolling Correlation')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='0.5 Correlation')
        ax3.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.7, label='-0.5 Correlation')
        
        ax3.set_title('Rolling Correlation (5-Year Window)', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylabel('Correlation', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for correlation analysis', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Rolling Correlation', fontweight='bold', fontsize=14)
    
    # Plot 4: Statistical Summary
    ax4.axis('off')
    
    summary_text = f"""
    📊 COMPARISON SUMMARY
    
    📈 Apple (AAPL):
    • Mean Annual Return: {aapl_stats['mean']:.2f}%
    • Annualized Volatility: {aapl_stats['std']:.2f}%
    • Median Return: {aapl_stats['median']:.2f}%
    • 68% Range: {aapl_stats['confidence_68_lower']:.1f}% to {aapl_stats['confidence_68_upper']:.1f}%
    
    📊 S&P 500 (^GSPC):
    • Mean Annual Return: {sp500_stats['mean']:.2f}%
    • Annualized Volatility: {sp500_stats['std']:.2f}%
    • Median Return: {sp500_stats['median']:.2f}%
    • 68% Range: {sp500_stats['confidence_68_lower']:.1f}% to {sp500_stats['confidence_68_upper']:.1f}%
    
    🎯 Key Insights:
    • Apple Outperformance: {aapl_stats['mean'] - sp500_stats['mean']:.1f}% per year
    • Apple Volatility: {aapl_stats['std'] - sp500_stats['std']:.1f}% higher
    • Risk-Adjusted Return: {(aapl_stats['mean']/aapl_stats['std']) - (sp500_stats['mean']/sp500_stats['std']):.2f} higher
    
    📅 Data Coverage:
    • Apple: {len(aapl_data)} years ({aapl_data['Year'].min()}-{aapl_data['Year'].max()})
    • S&P 500: {len(sp500_data)} years ({sp500_data['Year'].min()}-{sp500_data['Year'].max()})
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'apple_vs_sp500_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_focused_comparison_plot(aapl_data, sp500_data, aapl_stats, sp500_stats):
    """Create the main focused comparison plot."""
    print("📊 Creating focused comparison visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    returns_aapl = aapl_data['Annual_Return']
    returns_sp500 = sp500_data['Annual_Return']
    
    # Plot 1: Improved Returns Distribution Comparison
    x_min, x_max = -60, 120
    bins_aapl = np.linspace(x_min, x_max, 20)
    bins_sp500 = np.linspace(x_min, x_max, 15)
    
    ax1.hist(returns_aapl, bins=bins_aapl, alpha=0.7, edgecolor='black', 
             color='lightblue', density=True, label='Apple (AAPL) Returns')
    ax1.hist(returns_sp500, bins=bins_sp500, alpha=0.7, edgecolor='black', 
             color='lightcoral', density=True, label='S&P 500 (^GSPC) Returns')
    
    # Generate normal distribution curves
    x_range = np.linspace(x_min, x_max, 200)
    normal_aapl = stats.norm.pdf(x_range, aapl_stats['mean'], aapl_stats['std'])
    normal_sp500 = stats.norm.pdf(x_range, sp500_stats['mean'], sp500_stats['std'])
    
    ax1.plot(x_range, normal_aapl, 'b-', linewidth=3, label='AAPL Normal Distribution')
    ax1.plot(x_range, normal_sp500, 'r-', linewidth=3, label='S&P 500 Normal Distribution')
    
    # Add mean lines
    ax1.axvline(aapl_stats['mean'], color='blue', linestyle='--', linewidth=2, 
                label=f'AAPL Mean: {aapl_stats["mean"]:.1f}%')
    ax1.axvline(sp500_stats['mean'], color='red', linestyle='--', linewidth=2, 
                label=f'S&P 500 Mean: {sp500_stats["mean"]:.1f}%')
    
    ax1.set_title('Returns Distribution Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Annual Return (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Returns over time with improved styling
    ax2.plot(aapl_data['Year'], returns_aapl, 'o-', linewidth=2, markersize=6, 
             label='Apple (AAPL)', color='blue')
    ax2.plot(sp500_data['Year'], returns_sp500, 's-', linewidth=2, markersize=6, 
             label='S&P 500 (^GSPC)', color='red')
    
    # Add mean lines
    ax2.axhline(y=aapl_stats['mean'], color='blue', linestyle='--', alpha=0.7, 
                label=f'AAPL Mean: {aapl_stats["mean"]:.1f}%')
    ax2.axhline(y=sp500_stats['mean'], color='red', linestyle='--', alpha=0.7, 
                label=f'S&P 500 Mean: {sp500_stats["mean"]:.1f}%')
    
    ax2.set_title('Annual Returns Over Time', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Annual Return (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'apple_vs_sp500_focused.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_standardized_returns_analysis(aapl_data, sp500_data, aapl_stats, sp500_stats):
    """Create standardized returns comparison analysis."""
    print("📊 Creating standardized returns analysis...")
    
    returns_aapl = aapl_data['Annual_Return']
    returns_sp500 = sp500_data['Annual_Return']
    
    # Calculate z-scores (standardized returns)
    z_scores_aapl = (returns_aapl - aapl_stats['mean']) / aapl_stats['std']
    z_scores_sp500 = (returns_sp500 - sp500_stats['mean']) / sp500_stats['std']
    
    # Create figure with single subplot for standardized returns only
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot overlapping histogram of standardized returns (z-scores)
    z_min, z_max = -3, 4
    bins_z = np.linspace(z_min, z_max, 25)
    
    # Plot standardized returns histograms
    ax.hist(z_scores_aapl, bins=bins_z, alpha=0.7, edgecolor='black', 
            color='lightblue', density=True, label='Apple (AAPL) Z-Scores')
    ax.hist(z_scores_sp500, bins=bins_z, alpha=0.7, edgecolor='black', 
            color='lightcoral', density=True, label='S&P 500 (^GSPC) Z-Scores')
    
    # Add standard normal distribution overlay
    z_range = np.linspace(z_min, z_max, 200)
    standard_normal = stats.norm.pdf(z_range, 0, 1)
    ax.plot(z_range, standard_normal, 'k-', linewidth=3, label='Standard Normal Distribution')
    
    # Add reference lines at z = -1, 0, +1
    ax.axvline(-1, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Z = -1 (1 SD below mean)')
    ax.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Z = 0 (mean)')
    ax.axvline(1, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Z = +1 (1 SD above mean)')
    
    ax.set_title('Standardized Annual Return Comparison: AAPL vs S&P 500', fontsize=16, fontweight='bold')
    ax.set_xlabel('Z-Score (Standardized Annual Return)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_xlim(z_min, z_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    
    # Add summary box
    aapl_sharpe = aapl_stats['mean'] / aapl_stats['std']
    sp500_sharpe = sp500_stats['mean'] / sp500_stats['std']
    
    summary_text = f"""Standardization Summary:

Raw Statistics:
• AAPL Mean: {aapl_stats['mean']:.1f}% | Volatility: {aapl_stats['std']:.1f}%
• S&P 500 Mean: {sp500_stats['mean']:.1f}% | Volatility: {sp500_stats['std']:.1f}%

Sharpe Ratios (Mean/Volatility):
• AAPL Sharpe: {aapl_sharpe:.3f}
• S&P 500 Sharpe: {sp500_sharpe:.3f}

Z-Score Interpretation:
• Z = 0: Return equals the mean
• Z = +1: Return is 1 standard deviation above mean
• Z = -1: Return is 1 standard deviation below mean
• Z = +2: Return is 2 standard deviations above mean

Distribution Analysis:
• Both series now on same scale for direct comparison
• Standard normal curve shows theoretical distribution
• Outliers and distribution shapes clearly visible"""
    
    fig.text(0.75, 0.02, summary_text, fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'apple_vs_sp500_standardized.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print z-score statistics
    print(f"\n📊 Z-Score Statistics:")
    print(f"  AAPL Z-Score Range: {z_scores_aapl.min():.2f} to {z_scores_aapl.max():.2f}")
    print(f"  S&P 500 Z-Score Range: {z_scores_sp500.min():.2f} to {z_scores_sp500.max():.2f}")
    print(f"  AAPL Z-Score Mean: {z_scores_aapl.mean():.3f} (should be ~0)")
    print(f"  S&P 500 Z-Score Mean: {z_scores_sp500.mean():.3f} (should be ~0)")
    print(f"  AAPL Z-Score Std Dev: {z_scores_aapl.std():.3f} (should be ~1)")
    print(f"  S&P 500 Z-Score Std Dev: {z_scores_sp500.std():.3f} (should be ~1)")

def create_apple_vs_stock_comparison(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker):
    """Create comprehensive Apple vs individual stock comparison (similar to Apple vs S&P 500)."""
    print(f"📊 Creating Apple vs {stock_ticker} detailed comparison...")
    
    returns_aapl = aapl_data['Annual_Return']
    returns_stock = stock_data['Annual_Return']
    
    # Create main comparison visualization (4-panel layout)
    create_apple_vs_stock_main_comparison(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker)
    
    # Create focused comparison plot (2-panel layout)
    create_apple_vs_stock_focused_comparison(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker)
    
    # Create standardized returns analysis
    create_apple_vs_stock_standardized_analysis(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker)
    
    # Print detailed comparison analysis
    print_apple_vs_stock_analysis(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker)

def create_apple_vs_stock_main_comparison(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker):
    """Create the main 4-panel comparison visualization for Apple vs individual stock."""
    print(f"📊 Creating main comparison visualization for Apple vs {stock_ticker}...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    returns_aapl = aapl_data['Annual_Return']
    returns_stock = stock_data['Annual_Return']
    
    # Plot 1: Annual returns comparison
    x_min, x_max = -60, 120
    bins_aapl = np.linspace(x_min, x_max, 20)
    bins_stock = np.linspace(x_min, x_max, 15)
    
    ax1.hist(returns_aapl, bins=bins_aapl, alpha=0.7, edgecolor='black', 
             color='lightblue', density=True, label='Apple (AAPL) Returns')
    ax1.hist(returns_stock, bins=bins_stock, alpha=0.7, edgecolor='black', 
             color='lightcoral', density=True, label=f'{stock_ticker} Returns')
    
    # Add normal distribution overlays
    x_range = np.linspace(x_min, x_max, 200)
    normal_aapl = stats.norm.pdf(x_range, aapl_stats['mean'], aapl_stats['std'])
    normal_stock = stats.norm.pdf(x_range, stock_stats['mean'], stock_stats['std'])
    
    ax1.plot(x_range, normal_aapl, 'b-', linewidth=3, label='AAPL Normal Distribution')
    ax1.plot(x_range, normal_stock, 'r-', linewidth=3, label=f'{stock_ticker} Normal Distribution')
    
    # Add mean lines
    ax1.axvline(aapl_stats['mean'], color='blue', linestyle='--', linewidth=2, 
                label=f'AAPL Mean: {aapl_stats["mean"]:.1f}%')
    ax1.axvline(stock_stats['mean'], color='red', linestyle='--', linewidth=2, 
                label=f'{stock_ticker} Mean: {stock_stats["mean"]:.1f}%')
    
    ax1.set_title('Returns Distribution Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Annual Return (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Returns over time
    ax2.plot(aapl_data['Year'], returns_aapl, 'o-', linewidth=2, markersize=6, 
             label='Apple (AAPL)', color='blue')
    ax2.plot(stock_data['Year'], returns_stock, 's-', linewidth=2, markersize=6, 
             label=f'{stock_ticker}', color='red')
    
    ax2.axhline(y=aapl_stats['mean'], color='blue', linestyle='--', alpha=0.7, 
                label=f'AAPL Mean: {aapl_stats["mean"]:.1f}%')
    ax2.axhline(y=stock_stats['mean'], color='red', linestyle='--', alpha=0.7, 
                label=f'{stock_ticker} Mean: {stock_stats["mean"]:.1f}%')
    
    ax2.set_title('Annual Returns Over Time', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Annual Return (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Rolling Correlation
    combined_data = pd.merge(aapl_data, stock_data, on='Year', suffixes=('_AAPL', f'_{stock_ticker}'))
    if len(combined_data) >= 5:
        rolling_corr = combined_data['Annual_Return_AAPL'].rolling(window=5, min_periods=1).corr(combined_data[f'Annual_Return_{stock_ticker}'])
        
        ax3.plot(combined_data['Year'], rolling_corr, 'g-', linewidth=2, label='5-Year Rolling Correlation')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='0.5 Correlation')
        ax3.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.7, label='-0.5 Correlation')
        
        ax3.set_title('Rolling Correlation (5-Year Window)', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylabel('Correlation', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for correlation analysis', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Rolling Correlation', fontweight='bold', fontsize=14)
    
    # Plot 4: Statistical Summary
    ax4.axis('off')
    
    summary_text = f"""
    📊 COMPARISON SUMMARY
    
    📈 Apple (AAPL):
    • Mean Annual Return: {aapl_stats['mean']:.2f}%
    • Annualized Volatility: {aapl_stats['std']:.2f}%
    • Median Return: {aapl_stats['median']:.2f}%
    • 68% Range: {aapl_stats['confidence_68_lower']:.1f}% to {aapl_stats['confidence_68_upper']:.1f}%
    
    📊 {stock_ticker}:
    • Mean Annual Return: {stock_stats['mean']:.2f}%
    • Annualized Volatility: {stock_stats['std']:.2f}%
    • Median Return: {stock_stats['median']:.2f}%
    • 68% Range: {stock_stats['confidence_68_lower']:.1f}% to {stock_stats['confidence_68_upper']:.1f}%
    
    🎯 Key Insights:
    • Apple Outperformance: {aapl_stats['mean'] - stock_stats['mean']:.1f}% per year
    • Apple Volatility: {aapl_stats['std'] - stock_stats['std']:.1f}% higher
    • Risk-Adjusted Return: {(aapl_stats['mean']/aapl_stats['std']) - (stock_stats['mean']/stock_stats['std']):.2f} higher
    
    📅 Data Coverage:
    • Apple: {len(aapl_data)} years ({aapl_data['Year'].min()}-{aapl_data['Year'].max()})
    • {stock_ticker}: {len(stock_data)} years ({stock_data['Year'].min()}-{stock_data['Year'].max()})
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f'apple_vs_{stock_ticker.lower()}_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_apple_vs_stock_focused_comparison(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker):
    """Create focused 2-panel comparison for Apple vs individual stock."""
    print(f"📊 Creating focused comparison for Apple vs {stock_ticker}...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    returns_aapl = aapl_data['Annual_Return']
    returns_stock = stock_data['Annual_Return']
    
    # Plot 1: Improved Returns Distribution Comparison
    x_min, x_max = -60, 120
    bins_aapl = np.linspace(x_min, x_max, 20)
    bins_stock = np.linspace(x_min, x_max, 15)
    
    ax1.hist(returns_aapl, bins=bins_aapl, alpha=0.7, edgecolor='black', 
             color='lightblue', density=True, label='Apple (AAPL) Returns')
    ax1.hist(returns_stock, bins=bins_stock, alpha=0.7, edgecolor='black', 
             color='lightcoral', density=True, label=f'{stock_ticker} Returns')
    
    # Generate normal distribution curves
    x_range = np.linspace(x_min, x_max, 200)
    normal_aapl = stats.norm.pdf(x_range, aapl_stats['mean'], aapl_stats['std'])
    normal_stock = stats.norm.pdf(x_range, stock_stats['mean'], stock_stats['std'])
    
    ax1.plot(x_range, normal_aapl, 'b-', linewidth=3, label='AAPL Normal Distribution')
    ax1.plot(x_range, normal_stock, 'r-', linewidth=3, label=f'{stock_ticker} Normal Distribution')
    
    # Add mean lines
    ax1.axvline(aapl_stats['mean'], color='blue', linestyle='--', linewidth=2, 
                label=f'AAPL Mean: {aapl_stats["mean"]:.1f}%')
    ax1.axvline(stock_stats['mean'], color='red', linestyle='--', linewidth=2, 
                label=f'{stock_ticker} Mean: {stock_stats["mean"]:.1f}%')
    
    ax1.set_title('Returns Distribution Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Annual Return (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Returns over time with improved styling
    ax2.plot(aapl_data['Year'], returns_aapl, 'o-', linewidth=2, markersize=6, 
             label='Apple (AAPL)', color='blue')
    ax2.plot(stock_data['Year'], returns_stock, 's-', linewidth=2, markersize=6, 
             label=f'{stock_ticker}', color='red')
    
    # Add mean lines
    ax2.axhline(y=aapl_stats['mean'], color='blue', linestyle='--', alpha=0.7, 
                label=f'AAPL Mean: {aapl_stats["mean"]:.1f}%')
    ax2.axhline(y=stock_stats['mean'], color='red', linestyle='--', alpha=0.7, 
                label=f'{stock_ticker} Mean: {stock_stats["mean"]:.1f}%')
    
    ax2.set_title('Annual Returns Over Time', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Annual Return (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f'apple_vs_{stock_ticker.lower()}_focused.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_apple_vs_stock_standardized_analysis(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker):
    """Create standardized returns comparison analysis for Apple vs individual stock."""
    print(f"📊 Creating standardized returns analysis for Apple vs {stock_ticker}...")
    
    returns_aapl = aapl_data['Annual_Return']
    returns_stock = stock_data['Annual_Return']
    
    # Calculate z-scores (standardized returns)
    z_scores_aapl = (returns_aapl - aapl_stats['mean']) / aapl_stats['std']
    z_scores_stock = (returns_stock - stock_stats['mean']) / stock_stats['std']
    
    # Create figure with single subplot for standardized returns only
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot overlapping histogram of standardized returns (z-scores)
    z_min, z_max = -3, 4
    bins_z = np.linspace(z_min, z_max, 25)
    
    # Plot standardized returns histograms
    ax.hist(z_scores_aapl, bins=bins_z, alpha=0.7, edgecolor='black', 
            color='lightblue', density=True, label='Apple (AAPL) Z-Scores')
    ax.hist(z_scores_stock, bins=bins_z, alpha=0.7, edgecolor='black', 
            color='lightcoral', density=True, label=f'{stock_ticker} Z-Scores')
    
    # Add standard normal distribution overlay
    z_range = np.linspace(z_min, z_max, 200)
    standard_normal = stats.norm.pdf(z_range, 0, 1)
    ax.plot(z_range, standard_normal, 'k-', linewidth=3, label='Standard Normal Distribution')
    
    # Add reference lines at z = -1, 0, +1
    ax.axvline(-1, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Z = -1 (1 SD below mean)')
    ax.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Z = 0 (mean)')
    ax.axvline(1, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Z = +1 (1 SD above mean)')
    
    ax.set_title(f'Standardized Annual Return Comparison: AAPL vs {stock_ticker}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Z-Score (Standardized Annual Return)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_xlim(z_min, z_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    
    # Add summary box
    aapl_sharpe = aapl_stats['mean'] / aapl_stats['std']
    stock_sharpe = stock_stats['mean'] / stock_stats['std']
    
    summary_text = f"""Standardization Summary:

Raw Statistics:
• AAPL Mean: {aapl_stats['mean']:.1f}% | Volatility: {aapl_stats['std']:.1f}%
• {stock_ticker} Mean: {stock_stats['mean']:.1f}% | Volatility: {stock_stats['std']:.1f}%

Sharpe Ratios (Mean/Volatility):
• AAPL Sharpe: {aapl_sharpe:.3f}
• {stock_ticker} Sharpe: {stock_sharpe:.3f}

Z-Score Interpretation:
• Z = 0: Return equals the mean
• Z = +1: Return is 1 standard deviation above mean
• Z = -1: Return is 1 standard deviation below mean
• Z = +2: Return is 2 standard deviations above mean

Distribution Analysis:
• Both series now on same scale for direct comparison
• Standard normal curve shows theoretical distribution
• Outliers and distribution shapes clearly visible"""
    
    fig.text(0.75, 0.02, summary_text, fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f'apple_vs_{stock_ticker.lower()}_standardized.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print z-score statistics
    print(f"\n📊 Z-Score Statistics for Apple vs {stock_ticker}:")
    print(f"  AAPL Z-Score Range: {z_scores_aapl.min():.2f} to {z_scores_aapl.max():.2f}")
    print(f"  {stock_ticker} Z-Score Range: {z_scores_stock.min():.2f} to {z_scores_stock.max():.2f}")
    print(f"  AAPL Z-Score Mean: {z_scores_aapl.mean():.3f} (should be ~0)")
    print(f"  {stock_ticker} Z-Score Mean: {z_scores_stock.mean():.3f} (should be ~0)")
    print(f"  AAPL Z-Score Std Dev: {z_scores_aapl.std():.3f} (should be ~1)")
    print(f"  {stock_ticker} Z-Score Std Dev: {z_scores_stock.std():.3f} (should be ~1)")

def print_apple_vs_stock_analysis(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker):
    """Print detailed comparison analysis for Apple vs individual stock."""
    print(f"\n📊 APPLE VS {stock_ticker} COMPARISON ANALYSIS")
    print("=" * 60)
    
    print(f"\n📈 Performance Comparison:")
    print(f"  Apple Mean Annual Return: {aapl_stats['mean']:.2f}%")
    print(f"  {stock_ticker} Mean Annual Return: {stock_stats['mean']:.2f}%")
    print(f"  Apple Outperformance: {aapl_stats['mean'] - stock_stats['mean']:.2f}% per year")
    
    print(f"\n📊 Risk Comparison:")
    print(f"  Apple Volatility: {aapl_stats['std']:.2f}%")
    print(f"  {stock_ticker} Volatility: {stock_stats['std']:.2f}%")
    print(f"  Volatility Difference: {aapl_stats['std'] - stock_stats['std']:.2f}%")
    
    print(f"\n🎯 Risk-Adjusted Returns:")
    aapl_sharpe = aapl_stats['mean'] / aapl_stats['std']
    stock_sharpe = stock_stats['mean'] / stock_stats['std']
    print(f"  Apple Sharpe Ratio: {aapl_sharpe:.3f}")
    print(f"  {stock_ticker} Sharpe Ratio: {stock_sharpe:.3f}")
    print(f"  Sharpe Ratio Difference: {aapl_sharpe - stock_sharpe:.3f}")
    
    # Calculate correlation
    combined_data = pd.merge(aapl_data, stock_data, on='Year', suffixes=('_AAPL', f'_{stock_ticker}'))
    correlation = combined_data['Annual_Return_AAPL'].corr(combined_data[f'Annual_Return_{stock_ticker}'])
    print(f"\n🔗 Correlation Analysis:")
    print(f"  Correlation between Apple and {stock_ticker}: {correlation:.3f}")
    
    print(f"\n📅 Performance by Year:")
    print("Year   Apple    {}  Difference".format(stock_ticker))
    print("-" * 50)
    
    for _, row in combined_data.iterrows():
        apple_return = row['Annual_Return_AAPL']
        stock_return = row[f'Annual_Return_{stock_ticker}']
        difference = apple_return - stock_return
        
        # Add emoji indicators
        apple_emoji = "🟢" if apple_return > 0 else "🔴"
        stock_emoji = "🟢" if stock_return > 0 else "🔴"
        
        print(f"{row['Year']:.1f}  {apple_return:6.1f}% {apple_emoji} {stock_return:6.1f}% {stock_emoji} {difference:8.1f}%")
    
    print(f"\n🏆 Best Years:")
    print(f"  Apple: {aapl_stats['best_year']['Year']:.0f} ({aapl_stats['best_year']['Annual_Return']:.1f}%)")
    print(f"  {stock_ticker}: {stock_stats['best_year']['Year']:.0f} ({stock_stats['best_year']['Annual_Return']:.1f}%)")
    
    print(f"\n📉 Worst Years:")
    print(f"  Apple: {aapl_stats['worst_year']['Year']:.0f} ({aapl_stats['worst_year']['Annual_Return']:.1f}%)")
    print(f"  {stock_ticker}: {stock_stats['worst_year']['Year']:.0f} ({stock_stats['worst_year']['Annual_Return']:.1f}%)")

def create_mag7_comparison_visualization(mag7_data_dict, mag7_stats_dict):
    """Create comprehensive comparison visualization for all MAG7 stocks."""
    print("📊 Creating MAG7 comparison visualization...")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Colors for MAG7 stocks
    colors = ['#007AFF', '#00A4EF', '#FF9900', '#4285F4', '#76B900', '#CC0000', '#1877F2']
    
    # Plot 1: Annual returns comparison
    x_min, x_max = -80, 250
    bins = np.linspace(x_min, x_max, 30)
    
    for i, ticker in enumerate(MAG7_TICKERS):
        if ticker in mag7_data_dict:
            returns = mag7_data_dict[ticker]['Annual_Return']
            ax1.hist(returns, bins=bins, alpha=0.7, edgecolor='black', 
                    color=colors[i], density=True, label=f'{ticker} Returns')
    
    # Add normal distribution overlays
    x_range = np.linspace(x_min, x_max, 200)
    for i, ticker in enumerate(MAG7_TICKERS):
        if ticker in mag7_stats_dict:
            stats_dict = mag7_stats_dict[ticker]
            normal = stats.norm.pdf(x_range, stats_dict['mean'], stats_dict['std'])
            ax1.plot(x_range, normal, color=colors[i], linewidth=2, 
                    linestyle='--', alpha=0.8, label=f'{ticker} Normal')
    
    ax1.set_title('MAG7 Stocks: Annual Returns Distribution', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Annual Return (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Mean returns comparison
    tickers = list(mag7_stats_dict.keys())
    means = [mag7_stats_dict[ticker]['mean'] for ticker in tickers]
    stds = [mag7_stats_dict[ticker]['std'] for ticker in tickers]
    
    bars = ax2.bar(tickers, means, yerr=stds, capsize=5, alpha=0.7, color=colors[:len(tickers)])
    ax2.set_title('MAG7 Stocks: Mean Annual Returns with Volatility', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Annual Return (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Sharpe ratios
    sharpe_ratios = [mag7_stats_dict[ticker]['mean'] / mag7_stats_dict[ticker]['std'] 
                    for ticker in tickers]
    
    bars = ax3.bar(tickers, sharpe_ratios, alpha=0.7, color=colors[:len(tickers)])
    ax3.set_title('MAG7 Stocks: Sharpe Ratios (Mean/Volatility)', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Sharpe Ratio', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, sharpe in zip(bars, sharpe_ratios):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{sharpe:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Correlation heatmap
    # Calculate correlation matrix for all stocks
    all_returns = pd.DataFrame()
    for ticker in MAG7_TICKERS:
        if ticker in mag7_data_dict:
            all_returns[ticker] = mag7_data_dict[ticker]['Annual_Return']
    
    if not all_returns.empty:
        correlation_matrix = all_returns.corr()
        
        im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_title('MAG7 Stocks: Annual Returns Correlation Matrix', fontweight='bold', fontsize=14)
        
        # Add correlation values as text
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        ax4.set_xticks(range(len(correlation_matrix.columns)))
        ax4.set_yticks(range(len(correlation_matrix.columns)))
        ax4.set_xticklabels(correlation_matrix.columns)
        ax4.set_yticklabels(correlation_matrix.columns)
        
        # Add colorbar
        plt.colorbar(im, ax=ax4, label='Correlation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'mag7_comprehensive_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_individual_stock_analysis(ticker, annual_data, stats_dict):
    """Create individual analysis for each MAG7 stock."""
    print(f"📊 Creating individual analysis for {ticker}...")
    
    returns = annual_data['Annual_Return']
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Annual returns over time
    ax1.plot(annual_data['Year'], returns, 'o-', linewidth=2, markersize=6)
    ax1.axhline(y=stats_dict['mean'], color='red', linestyle='--', 
                label=f'Mean: {stats_dict["mean"]:.1f}%')
    ax1.fill_between(annual_data['Year'], 
                     stats_dict['confidence_68_lower'], 
                     stats_dict['confidence_68_upper'], 
                     alpha=0.3, color='red', label='±1 Std Dev')
    
    ax1.set_title(f'{ticker}: Annual Returns Over Time', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Annual Return (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Returns distribution
    x_min, x_max = returns.min() - 10, returns.max() + 10
    bins = np.linspace(x_min, x_max, 20)
    
    ax2.hist(returns, bins=bins, alpha=0.7, edgecolor='black', density=True)
    
    # Add normal distribution overlay
    x_range = np.linspace(x_min, x_max, 200)
    normal = stats.norm.pdf(x_range, stats_dict['mean'], stats_dict['std'])
    ax2.plot(x_range, normal, 'r-', linewidth=2, label='Normal Distribution')
    
    ax2.axvline(stats_dict['mean'], color='red', linestyle='--', 
                label=f'Mean: {stats_dict["mean"]:.1f}%')
    
    ax2.set_title(f'{ticker}: Returns Distribution', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Annual Return (%)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot
    ax3.boxplot(returns, patch_artist=True)
    ax3.set_title(f'{ticker}: Returns Box Plot', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Annual Return (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistical summary
    ax4.axis('off')
    
    summary_text = f"""
    📊 {ticker} STATISTICAL SUMMARY
    
    📈 Performance:
    • Mean Annual Return: {stats_dict['mean']:.2f}%
    • Median Return: {stats_dict['median']:.2f}%
    • Volatility (Std Dev): {stats_dict['std']:.2f}%
    • Sharpe Ratio: {stats_dict['mean']/stats_dict['std']:.3f}
    
    📊 Distribution:
    • Skewness: {stats_dict['skewness']:.3f}
    • Kurtosis: {stats_dict['kurtosis']:.3f}
    • 68% Range: {stats_dict['confidence_68_lower']:.1f}% to {stats_dict['confidence_68_upper']:.1f}%
    
    🏆 Best Year: {stats_dict['best_year']['Year']:.0f} ({stats_dict['best_year']['Annual_Return']:.1f}%)
    📉 Worst Year: {stats_dict['worst_year']['Year']:.0f} ({stats_dict['worst_year']['Annual_Return']:.1f}%)
    
    📅 Data Coverage: {stats_dict['total_years']} years
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f'{ticker}_individual_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# ANALYSIS SUMMARY FUNCTIONS
# ============================================================================

def print_comparison_analysis(aapl_data, sp500_data, aapl_stats, sp500_stats):
    """Print detailed comparison analysis."""
    print("\n📊 APPLE VS S&P 500 COMPARISON ANALYSIS")
    print("=" * 60)
    
    print(f"\n📈 Performance Comparison:")
    print(f"  Apple Mean Annual Return: {aapl_stats['mean']:.2f}%")
    print(f"  S&P 500 Mean Annual Return: {sp500_stats['mean']:.2f}%")
    print(f"  Apple Outperformance: {aapl_stats['mean'] - sp500_stats['mean']:.2f}% per year")
    
    print(f"\n📊 Risk Comparison:")
    print(f"  Apple Volatility: {aapl_stats['std']:.2f}%")
    print(f"  S&P 500 Volatility: {sp500_stats['std']:.2f}%")
    print(f"  Volatility Difference: {aapl_stats['std'] - sp500_stats['std']:.2f}%")
    
    print(f"\n🎯 Risk-Adjusted Returns:")
    aapl_sharpe = aapl_stats['mean'] / aapl_stats['std']
    sp500_sharpe = sp500_stats['mean'] / sp500_stats['std']
    print(f"  Apple Sharpe Ratio: {aapl_sharpe:.3f}")
    print(f"  S&P 500 Sharpe Ratio: {sp500_sharpe:.3f}")
    print(f"  Sharpe Ratio Difference: {aapl_sharpe - sp500_sharpe:.3f}")
    
    # Calculate correlation
    combined_data = pd.merge(aapl_data, sp500_data, on='Year', suffixes=('_AAPL', '_SP500'))
    correlation = combined_data['Annual_Return_AAPL'].corr(combined_data['Annual_Return_SP500'])
    print(f"\n🔗 Correlation Analysis:")
    print(f"  Correlation between Apple and S&P 500: {correlation:.3f}")
    
    print(f"\n📅 Performance by Year:")
    print("Year   Apple    S&P 500  Difference")
    print("-" * 40)
    
    for _, row in combined_data.iterrows():
        apple_return = row['Annual_Return_AAPL']
        sp500_return = row['Annual_Return_SP500']
        difference = apple_return - sp500_return
        
        # Add emoji indicators
        apple_emoji = "🟢" if apple_return > 0 else "🔴"
        sp500_emoji = "🟢" if sp500_return > 0 else "🔴"
        
        print(f"{row['Year']:.1f}  {apple_return:6.1f}% {apple_emoji} {sp500_return:6.1f}% {sp500_emoji} {difference:8.1f}%")
    
    print(f"\n🏆 Best Years:")
    print(f"  Apple: {aapl_stats['best_year']['Year']:.0f} ({aapl_stats['best_year']['Annual_Return']:.1f}%)")
    print(f"  S&P 500: {sp500_stats['best_year']['Year']:.0f} ({sp500_stats['best_year']['Annual_Return']:.1f}%)")
    
    print(f"\n📉 Worst Years:")
    print(f"  Apple: {aapl_stats['worst_year']['Year']:.0f} ({aapl_stats['worst_year']['Annual_Return']:.1f}%)")
    print(f"  S&P 500: {sp500_stats['worst_year']['Year']:.0f} ({sp500_stats['worst_year']['Annual_Return']:.1f}%)")

# ============================================================================
# ENHANCED ROLLING CORRELATION ANALYSIS
# ============================================================================

def create_enhanced_rolling_correlation_analysis():
    """Create enhanced rolling correlation analysis with multiple time windows and macro events for Apple vs high correlation stocks."""
    print("📊 Creating enhanced rolling correlation analysis for Apple vs high correlation stocks...")
    
    # Connect to database
    engine, inspector = connect_to_database()
    
    # Load Apple data
    aapl_data = load_stock_data(engine, inspector, 'AAPL')
    if aapl_data is None:
        print("❌ Could not load Apple data")
        return
    
    # Load S&P 500 data
    sp500_tickers = ['^GSPC', 'GSPC', 'SP500', 'SPX']
    sp500_data = None
    sp500_ticker = None
    
    for ticker in sp500_tickers:
        data = load_stock_data(engine, inspector, ticker)
        if data is not None:
            sp500_data = data
            sp500_ticker = ticker
            print(f"✅ Found S&P 500 data with ticker: {ticker}")
            break
    
    if sp500_data is None:
        print("❌ Could not load S&P 500 data")
        return
    
    # Load high correlation stocks (GOOGL, MSFT)
    high_corr_stocks = {}
    for ticker in ['GOOGL', 'MSFT']:
        data = load_stock_data(engine, inspector, ticker)
        if data is not None:
            high_corr_stocks[ticker] = data
            print(f"✅ Loaded {ticker} data for correlation analysis")
    
    # Create rolling correlation analysis for Apple vs S&P 500
    print("\n🍎 Creating Apple vs S&P 500 rolling correlation analysis...")
    create_rolling_correlation_analysis(aapl_data, sp500_data, 'AAPL', 'S&P 500')
    
    # Create rolling correlation analysis for Apple vs high correlation stocks
    for ticker, data in high_corr_stocks.items():
        print(f"\n🍎 Creating Apple vs {ticker} rolling correlation analysis...")
        create_rolling_correlation_analysis(aapl_data, data, 'AAPL', ticker)

def create_rolling_correlation_analysis(stock_data, sp500_data, stock_ticker, benchmark_ticker):
    """Create comprehensive rolling correlation analysis with multiple time windows."""
    print(f"📊 Creating rolling correlation analysis for {stock_ticker} vs {benchmark_ticker}...")
    
    # Get close prices
    stock_close_col = [col for col in stock_data.columns if 'Close' in col][0]
    sp500_close_col = [col for col in sp500_data.columns if 'Close' in col][0]
    
    stock_prices = stock_data[stock_close_col]
    sp500_prices = sp500_data[sp500_close_col]
    
    # Calculate daily returns
    stock_returns = stock_prices.pct_change() * 100
    sp500_returns = sp500_prices.pct_change() * 100
    
    # Align data by date
    combined_returns = pd.DataFrame({
        f'{stock_ticker}_Return': stock_returns,
        f'{benchmark_ticker}_Return': sp500_returns
    }).dropna()
    
    if len(combined_returns) < 252:  # Need at least 1 year of data
        print(f"❌ Insufficient data for {stock_ticker} vs {benchmark_ticker}")
        return
    
    # Calculate rolling correlations for different time windows
    windows = {
        '3 Months': 63,    # ~3 months of trading days
        '6 Months': 126,   # ~6 months of trading days
        '1 Year': 252,     # ~1 year of trading days
        '3 Years': 756     # ~3 years of trading days
    }
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'{stock_ticker} vs {benchmark_ticker}: Rolling Correlations Across Different Time Windows', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (window_name, window_days) in enumerate(windows.items()):
        ax = axes[i]
        
        # Calculate rolling correlation
        rolling_corr = combined_returns[f'{stock_ticker}_Return'].rolling(
            window=window_days, min_periods=window_days//2
        ).corr(combined_returns[f'{benchmark_ticker}_Return'])
        
        # Plot the correlation
        ax.plot(combined_returns.index, rolling_corr, 'b-', linewidth=2, 
                label=f'{window_name} Rolling Correlation')
        
        # Add reference lines
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, 
                  label='Strong Positive Correlation (0.7)')
        ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.7, 
                  label='No Correlation (0.0)')
        ax.axhline(y=-0.7, color='red', linestyle='--', alpha=0.7, 
                  label='Strong Negative Correlation (-0.7)')
        
        # Add major macro events
        add_macro_events(ax, combined_returns.index)
        
        # Formatting
        ax.set_title(f'{window_name} Rolling Correlation', fontweight='bold', fontsize=12)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Correlation', fontsize=10)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f'{stock_ticker}_vs_{benchmark_ticker.replace(" ", "_")}_rolling_correlations.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print correlation statistics
    print(f"\n📊 {stock_ticker} vs {benchmark_ticker} Rolling Correlation Statistics:")
    for window_name, window_days in windows.items():
        rolling_corr = combined_returns[f'{stock_ticker}_Return'].rolling(
            window=window_days, min_periods=window_days//2
        ).corr(combined_returns[f'{benchmark_ticker}_Return'])
        
        print(f"  {window_name}:")
        print(f"    Mean Correlation: {rolling_corr.mean():.3f}")
        print(f"    Std Deviation: {rolling_corr.std():.3f}")
        print(f"    Min Correlation: {rolling_corr.min():.3f}")
        print(f"    Max Correlation: {rolling_corr.max():.3f}")

def add_macro_events(ax, date_index):
    """Add major macro events as vertical lines on the correlation plot."""
    events = [
        ('2008-09-15', 'Lehman Crisis', 'red'),
        ('2008-10-03', 'TARP Bailout', 'orange'),
        ('2010-05-06', 'Flash Crash', 'purple'),
        ('2011-08-05', 'US Debt Downgrade', 'brown'),
        ('2012-09-13', 'QE3 Announcement', 'green'),
        ('2015-08-24', 'China Market Crash', 'red'),
        ('2016-11-08', 'US Election', 'blue'),
        ('2018-02-05', 'Volatility Spike', 'orange'),
        ('2018-10-10', 'Tech Sell-off', 'purple'),
        ('2020-03-16', 'COVID-19 Crash', 'red'),
        ('2020-11-09', 'Vaccine News', 'green'),
        ('2022-02-24', 'Russia-Ukraine War', 'brown'),
        ('2022-03-16', 'Fed Rate Hike', 'blue'),
        ('2023-03-10', 'SVB Collapse', 'red'),
        ('2023-11-30', 'AI Rally Continues', 'green')
    ]
    
    # Get the y-axis limits for better text positioning
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # Normalize timezone handling - make both timezone-naive for comparison
    date_index_naive = date_index.tz_localize(None) if date_index.tz is not None else date_index
    
    for event_date, event_name, color in events:
        try:
            event_dt = pd.to_datetime(event_date).tz_localize(None)  # Ensure timezone-naive
            # Check if event is within the data range
            if event_dt >= date_index_naive.min() and event_dt <= date_index_naive.max():
                # Draw vertical line - make it more visible
                ax.axvline(x=event_dt, color=color, linestyle='--', alpha=0.8, linewidth=2)
                
                # Add text label with better positioning
                text_y = y_max - (y_range * 0.1)  # Position in top 10% of plot
                ax.text(event_dt, text_y, event_name, 
                       rotation=90, fontsize=9, color=color, alpha=0.9,
                       ha='right', va='top', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                
                print(f"✅ Added macro event: {event_name} on {event_date}")
        except Exception as e:
            print(f"❌ Error adding event {event_name}: {e}")
            continue

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function for FMP stock analysis with 35+ years of historical data."""
    print("🚀 FMP Stock Analysis Module (35+ Years of Data)")
    print("=" * 60)
    print("📊 Analyzing comprehensive historical data from FMP API")
    print("📅 Date range: 1990-2025 (35+ years)")
    print("🎯 MAG7 stocks + S&P 500 analysis")
    print("=" * 60)
    
    # Run Apple vs S&P 500 analysis
    print("\n🍎 Running Apple vs S&P 500 analysis (35+ years)...")
    apple_vs_sp500_results = create_apple_vs_sp500_analysis()
    
    # Run MAG7 analysis
    print("\n🚀 Running MAG7 stocks analysis (complete historical coverage)...")
    mag7_results = create_mag7_analysis()
    
    # Run enhanced rolling correlation analysis
    print("\n📊 Running enhanced rolling correlation analysis...")
    create_enhanced_rolling_correlation_analysis()
    
    print("\n✅ FMP Analysis complete!")
    print(f"\n📊 Check the 'graphs' directory for comprehensive visualizations")
    print(f"📈 Analysis covers multiple market cycles and economic events")
    print(f"🎯 Professional-grade insights from FMP data")

if __name__ == "__main__":
    main()
