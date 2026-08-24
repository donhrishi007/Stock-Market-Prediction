"""
Stock Analysis Module - STEP 2 (LOG RETURNS VERSION - FMP Data Optimized)
========================================================================

📊 EXECUTION ORDER: RUN THIS FILE SECOND (after FMP data download)!

This module handles comprehensive analysis of FMP stock data using LOG TRANSFORMED RETURNS:
- Apple vs S&P 500 comparison analysis (log returns, 35+ years of data)
- MAG7 stocks analysis and comparison (log returns, complete historical coverage)
- Annual log returns calculations with extended historical analysis
- Statistical analysis and visualizations optimized for FMP data
- Correlation analysis across multiple market cycles

Key Benefits of Log Returns:
- Symmetric distribution (better for statistical analysis)
- Time-additive property (log returns can be summed over time)
- Better handling of extreme values
- More appropriate for financial modeling
- Enhanced analysis with 35+ years of FMP professional-grade data

Features:
- Optimized for 35+ years of historical data (1990-2025)
- Enhanced log returns analysis with FMP professional-grade data
- Comprehensive market cycle analysis with log transformations
- Advanced statistical metrics and visualizations
- Professional-grade insights for stock prediction modeling

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
GRAPHS_DIR = os.path.join(os.path.dirname(__file__), 'graphs', 'log_returns_analysis')
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def connect_to_database():
    """Connect to the SQLite database containing FMP stock data."""
    print("🔌 Connecting to FMP database for log returns analysis...")
    
    db_path = os.path.join(os.path.dirname(__file__), DB_NAME)
    engine = create_engine(f'sqlite:///{db_path}')
    inspector = inspect(engine)
    
    # Get table information
    tables = inspector.get_table_names()
    print(f"✅ Connected to database: {DB_NAME}")
    print(f"📊 Found {len(tables)} tables with FMP data for log returns analysis")
    
    return engine, inspector

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_stock_data(engine, inspector, ticker):
    """Load FMP stock data from database for log returns analysis."""
    print(f"📊 Loading {ticker} FMP price data for log returns analysis...")
    
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
        print(f"📊 Ready for log returns transformation")
        
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
# LOG RETURNS CALCULATION
# ============================================================================

def calculate_annual_log_returns(df, ticker):
    """Calculate annual log returns for a stock.
    
    Log returns are calculated as: log(1 + return/100)
    This transforms percentage returns to log space for better statistical properties.
    """
    print(f"📈 Calculating {ticker} annual log returns...")
    
    # Find the Close column (it might be named Close_TICKER)
    close_col = [col for col in df.columns if 'Close' in col][0]
    annual_prices = df[close_col].resample('Y').last()
    
    # Calculate percentage returns first
    percentage_returns = annual_prices.pct_change() * 100
    
    # Convert to log returns: log(1 + return/100)
    # This handles the transformation from percentage to log space
    log_returns = np.log(1 + percentage_returns / 100) * 100  # Keep in percentage scale for interpretation
    
    # Create DataFrame with Year and Log_Return columns
    annual_data = pd.DataFrame({
        'Year': log_returns.index.year,
        'Log_Return': log_returns.values
    })
    
    # Remove NaN values (first year will have NaN)
    annual_data = annual_data.dropna()
    
    print(f"📊 Calculated {len(annual_data)} years of {ticker} annual log returns")
    print(f"📅 Year range: {annual_data['Year'].min()} to {annual_data['Year'].max()}")
    
    return annual_data

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def calculate_log_returns_statistics(annual_data, ticker):
    """Calculate comprehensive statistics for annual log returns.
    
    Log returns have different statistical properties:
    - More symmetric distribution
    - Better for time series analysis
    - Can be summed over time periods
    """
    print(f"📊 Calculating {ticker} log returns statistics...")
    
    log_returns = annual_data['Log_Return']
    
    # Basic statistics
    mean_log_return = log_returns.mean()
    std_log_return = log_returns.std()
    median_log_return = log_returns.median()
    
    # Percentiles
    percentiles = log_returns.quantile([0.05, 0.25, 0.75, 0.95])
    
    # Confidence intervals (68% = ±1 std dev)
    confidence_68_lower = mean_log_return - std_log_return
    confidence_68_upper = mean_log_return + std_log_return
    
    # Skewness and kurtosis (should be closer to normal for log returns)
    skewness = log_returns.skew()
    kurtosis = log_returns.kurtosis()
    
    # Best and worst years
    best_year = annual_data.loc[log_returns.idxmax()]
    worst_year = annual_data.loc[log_returns.idxmin()]
    
    stats_dict = {
        'mean': mean_log_return,
        'std': std_log_return,
        'median': median_log_return,
        'percentiles': percentiles,
        'confidence_68_lower': confidence_68_lower,
        'confidence_68_upper': confidence_68_upper,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'best_year': best_year,
        'worst_year': worst_year,
        'total_years': len(log_returns)
    }
    
    print(f"📈 {ticker} Mean Annual Log Return: {mean_log_return:.2f}%")
    print(f"📊 {ticker} Log Return Volatility (Std Dev): {std_log_return:.2f}%")
    print(f"📉 {ticker} Median Log Return: {median_log_return:.2f}%")
    
    # Volatility analysis
    print(f"🎯 {ticker} Volatility Analysis:")
    print(f"  📊 Annual Volatility: {std_log_return:.2f}%")
    print(f"  📈 Volatility Range: {percentiles[0.05]:.2f}% to {percentiles[0.95]:.2f}%")
    print(f"  📉 68% Confidence: {confidence_68_lower:.2f}% to {confidence_68_upper:.2f}%")
    print(f"  🎲 Skewness: {skewness:.3f} (0 = symmetric)")
    print(f"  📊 Kurtosis: {kurtosis:.3f} (3 = normal)")
    
    return stats_dict

# ============================================================================
# VOLATILITY ANALYSIS FUNCTIONS
# ============================================================================

def create_volatility_analysis_visualization(aapl_data, sp500_data, aapl_stats, sp500_stats):
    """Create comprehensive volatility analysis visualization."""
    print("📊 Creating volatility analysis visualization...")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Volatility Analysis: Apple vs S&P 500 (Log Returns)', fontsize=16, fontweight='bold')
    
    # Plot 1: Volatility Comparison (Bar Chart)
    stocks = ['Apple', 'S&P 500']
    volatilities = [aapl_stats['std'], sp500_stats['std']]
    colors = ['#007AFF', '#FF6B6B']
    
    bars = ax1.bar(stocks, volatilities, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Annual Volatility Comparison', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Volatility (Standard Deviation %)', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, vol in zip(bars, volatilities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{vol:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Plot 2: Rolling Volatility (30-day window)
    aapl_close = aapl_data['Close_AAPL'] if 'Close_AAPL' in aapl_data.columns else aapl_data.iloc[:, 1]
    sp500_close = sp500_data['Close_SP500'] if 'Close_SP500' in sp500_data.columns else sp500_data.iloc[:, 1]
    
    # Calculate daily log returns
    aapl_daily_log_returns = np.log(aapl_close / aapl_close.shift(1)) * 100
    sp500_daily_log_returns = np.log(sp500_close / sp500_close.shift(1)) * 100
    
    # Calculate 30-day rolling volatility
    aapl_rolling_vol = aapl_daily_log_returns.rolling(window=30).std() * np.sqrt(252)  # Annualized
    sp500_rolling_vol = sp500_daily_log_returns.rolling(window=30).std() * np.sqrt(252)  # Annualized
    
    ax2.plot(aapl_rolling_vol.index, aapl_rolling_vol, label='Apple', color='#007AFF', linewidth=2)
    ax2.plot(sp500_rolling_vol.index, sp500_rolling_vol, label='S&P 500', color='#FF6B6B', linewidth=2)
    ax2.set_title('30-Day Rolling Volatility (Annualized)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Volatility (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volatility Distribution (Histogram)
    ax3.hist(aapl_data['Log_Return'], bins=20, alpha=0.7, label='Apple', color='#007AFF', density=True)
    ax3.hist(sp500_data['Log_Return'], bins=20, alpha=0.7, label='S&P 500', color='#FF6B6B', density=True)
    ax3.set_title('Log Returns Distribution (Volatility Profile)', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Log Returns (%)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add volatility lines
    ax3.axvline(aapl_stats['mean'], color='#007AFF', linestyle='--', alpha=0.8, label=f"Apple Mean: {aapl_stats['mean']:.2f}%")
    ax3.axvline(sp500_stats['mean'], color='#FF6B6B', linestyle='--', alpha=0.8, label=f"S&P 500 Mean: {sp500_stats['mean']:.2f}%")
    
    # Plot 4: Risk-Return Scatter
    ax4.scatter(aapl_stats['std'], aapl_stats['mean'], s=200, color='#007AFF', alpha=0.8, 
               label='Apple', edgecolor='black', linewidth=2)
    ax4.scatter(sp500_stats['std'], sp500_stats['mean'], s=200, color='#FF6B6B', alpha=0.8, 
               label='S&P 500', edgecolor='black', linewidth=2)
    
    ax4.set_title('Risk-Return Profile', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Volatility (Risk) %', fontsize=12)
    ax4.set_ylabel('Expected Return %', fontsize=12)
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Add labels for each point
    ax4.annotate('Apple', (aapl_stats['std'], aapl_stats['mean']), 
                xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold')
    ax4.annotate('S&P 500', (sp500_stats['std'], sp500_stats['mean']), 
                xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'volatility_analysis_comprehensive.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Volatility analysis visualization created!")

# ============================================================================
# APPLE VS S&P 500 LOG RETURNS ANALYSIS
# ============================================================================

def create_apple_vs_sp500_log_analysis():
    """Create comprehensive Apple vs S&P 500 analysis using log returns."""
    print("🍎 Creating Apple vs S&P 500 log returns analysis...")
    
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
    
    # Calculate annual log returns
    aapl_annual = calculate_annual_log_returns(aapl_data, 'AAPL')
    sp500_annual = calculate_annual_log_returns(sp500_data, sp500_ticker)
    
    # Calculate statistics
    aapl_stats = calculate_log_returns_statistics(aapl_annual, 'AAPL')
    sp500_stats = calculate_log_returns_statistics(sp500_annual, sp500_ticker)
    
    # Create visualizations
    create_log_returns_comparison_visualization(aapl_annual, sp500_annual, aapl_stats, sp500_stats)
    create_log_returns_focused_plot(aapl_annual, sp500_annual, aapl_stats, sp500_stats)
    create_log_returns_standardized_analysis(aapl_annual, sp500_annual, aapl_stats, sp500_stats)
    
    # Create volatility analysis
    create_volatility_analysis_visualization(aapl_data, sp500_data, aapl_stats, sp500_stats)
    
    # Print analysis
    print_log_returns_comparison_analysis(aapl_annual, sp500_annual, aapl_stats, sp500_stats)
    
    return aapl_annual, sp500_annual, aapl_stats, sp500_stats

# ============================================================================
# MAG7 LOG RETURNS ANALYSIS
# ============================================================================

def create_mag7_log_analysis():
    """Create comprehensive MAG7 stocks analysis using log returns."""
    print("🚀 Creating MAG7 stocks log returns analysis...")
    
    # Connect to database
    engine, inspector = connect_to_database()
    
    # Load all MAG7 data
    mag7_data = load_all_mag7_data(engine, inspector)
    
    if not mag7_data:
        print("❌ No MAG7 data loaded")
        return
    
    # Calculate annual log returns for all stocks
    mag7_annual_data = {}
    mag7_stats = {}
    
    for ticker in MAG7_TICKERS:
        if ticker in mag7_data:
            annual_data = calculate_annual_log_returns(mag7_data[ticker], ticker)
            stats = calculate_log_returns_statistics(annual_data, ticker)
            
            mag7_annual_data[ticker] = annual_data
            mag7_stats[ticker] = stats
    
    # Create Apple vs each MAG7 stock detailed comparisons
    if 'AAPL' in mag7_annual_data:
        aapl_data = mag7_annual_data['AAPL']
        aapl_stats = mag7_stats['AAPL']
        
        for ticker in MAG7_TICKERS:
            if ticker != 'AAPL' and ticker in mag7_annual_data:
                print(f"\n🍎 Creating Apple vs {ticker} log returns comparison...")
                create_apple_vs_stock_log_comparison(aapl_data, mag7_annual_data[ticker], 
                                                   aapl_stats, mag7_stats[ticker], ticker)
    
    # Create MAG7 comparison visualization
    if len(mag7_annual_data) > 1:
        create_mag7_log_comparison_visualization(mag7_annual_data, mag7_stats)
    
    return mag7_annual_data, mag7_stats

# ============================================================================
# VISUALIZATION FUNCTIONS FOR LOG RETURNS
# ============================================================================

def create_log_returns_comparison_visualization(aapl_data, sp500_data, aapl_stats, sp500_stats):
    """Create the main comparison visualization for log returns."""
    print("📊 Creating log returns comparison visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    log_returns_aapl = aapl_data['Log_Return']
    log_returns_sp500 = sp500_data['Log_Return']
    
    # Plot 1: Log returns distribution comparison
    x_min, x_max = -80, 100  # Adjusted range for log returns
    bins_aapl = np.linspace(x_min, x_max, 20)
    bins_sp500 = np.linspace(x_min, x_max, 15)
    
    ax1.hist(log_returns_aapl, bins=bins_aapl, alpha=0.7, edgecolor='black', 
             color='lightblue', density=True, label='Apple (AAPL) Log Returns')
    ax1.hist(log_returns_sp500, bins=bins_sp500, alpha=0.7, edgecolor='black', 
             color='lightcoral', density=True, label='S&P 500 (^GSPC) Log Returns')
    
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
    
    ax1.set_title('Log Returns Distribution Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Annual Log Return (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Log returns over time
    ax2.plot(aapl_data['Year'], log_returns_aapl, 'o-', linewidth=2, markersize=6, 
             label='Apple (AAPL)', color='blue')
    ax2.plot(sp500_data['Year'], log_returns_sp500, 's-', linewidth=2, markersize=6, 
             label='S&P 500 (^GSPC)', color='red')
    
    ax2.axhline(y=aapl_stats['mean'], color='blue', linestyle='--', alpha=0.7, 
                label=f'AAPL Mean: {aapl_stats["mean"]:.1f}%')
    ax2.axhline(y=sp500_stats['mean'], color='red', linestyle='--', alpha=0.7, 
                label=f'S&P 500 Mean: {sp500_stats["mean"]:.1f}%')
    
    ax2.set_title('Annual Log Returns Over Time', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Annual Log Return (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Rolling Correlation
    combined_data = pd.merge(aapl_data, sp500_data, on='Year', suffixes=('_AAPL', '_SP500'))
    if len(combined_data) >= 5:
        rolling_corr = combined_data['Log_Return_AAPL'].rolling(window=5, min_periods=1).corr(combined_data['Log_Return_SP500'])
        
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
    📊 LOG RETURNS COMPARISON SUMMARY
    
    📈 Apple (AAPL) Log Returns:
    • Mean Annual Log Return: {aapl_stats['mean']:.2f}%
    • Log Return Volatility: {aapl_stats['std']:.2f}%
    • Median Log Return: {aapl_stats['median']:.2f}%
    • 68% Range: {aapl_stats['confidence_68_lower']:.1f}% to {aapl_stats['confidence_68_upper']:.1f}%
    
    📊 S&P 500 (^GSPC) Log Returns:
    • Mean Annual Log Return: {sp500_stats['mean']:.2f}%
    • Log Return Volatility: {sp500_stats['std']:.2f}%
    • Median Log Return: {sp500_stats['median']:.2f}%
    • 68% Range: {sp500_stats['confidence_68_lower']:.1f}% to {sp500_stats['confidence_68_upper']:.1f}%
    
    🎯 Key Insights (Log Returns):
    • Apple Log Outperformance: {aapl_stats['mean'] - sp500_stats['mean']:.1f}% per year
    • Apple Log Volatility: {aapl_stats['std'] - sp500_stats['std']:.1f}% higher
    • Risk-Adjusted Log Return: {(aapl_stats['mean']/aapl_stats['std']) - (sp500_stats['mean']/sp500_stats['std']):.2f} higher
    
    📅 Data Coverage:
    • Apple: {len(aapl_data)} years ({aapl_data['Year'].min()}-{aapl_data['Year'].max()})
    • S&P 500: {len(sp500_data)} years ({sp500_data['Year'].min()}-{sp500_data['Year'].max()})
    
    💡 Log Returns Benefits:
    • More symmetric distribution
    • Better for statistical analysis
    • Time-additive property
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'apple_vs_sp500_log_returns_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_log_returns_focused_plot(aapl_data, sp500_data, aapl_stats, sp500_stats):
    """Create focused 2-panel comparison for log returns."""
    print("📊 Creating focused log returns visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    log_returns_aapl = aapl_data['Log_Return']
    log_returns_sp500 = sp500_data['Log_Return']
    
    # Plot 1: Log returns distribution comparison
    x_min, x_max = -80, 100
    bins_aapl = np.linspace(x_min, x_max, 20)
    bins_sp500 = np.linspace(x_min, x_max, 15)
    
    ax1.hist(log_returns_aapl, bins=bins_aapl, alpha=0.7, edgecolor='black', 
             color='lightblue', density=True, label='Apple (AAPL) Log Returns')
    ax1.hist(log_returns_sp500, bins=bins_sp500, alpha=0.7, edgecolor='black', 
             color='lightcoral', density=True, label='S&P 500 (^GSPC) Log Returns')
    
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
    
    ax1.set_title('Log Returns Distribution Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Annual Log Return (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Log returns over time
    ax2.plot(aapl_data['Year'], log_returns_aapl, 'o-', linewidth=2, markersize=6, 
             label='Apple (AAPL)', color='blue')
    ax2.plot(sp500_data['Year'], log_returns_sp500, 's-', linewidth=2, markersize=6, 
             label='S&P 500 (^GSPC)', color='red')
    
    # Add mean lines
    ax2.axhline(y=aapl_stats['mean'], color='blue', linestyle='--', alpha=0.7, 
                label=f'AAPL Mean: {aapl_stats["mean"]:.1f}%')
    ax2.axhline(y=sp500_stats['mean'], color='red', linestyle='--', alpha=0.7, 
                label=f'S&P 500 Mean: {sp500_stats["mean"]:.1f}%')
    
    ax2.set_title('Annual Log Returns Over Time', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Annual Log Return (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'apple_vs_sp500_log_returns_focused.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_log_returns_standardized_analysis(aapl_data, sp500_data, aapl_stats, sp500_stats):
    """Create standardized log returns comparison analysis."""
    print("📊 Creating standardized log returns analysis...")
    
    log_returns_aapl = aapl_data['Log_Return']
    log_returns_sp500 = sp500_data['Log_Return']
    
    # Calculate z-scores (standardized log returns)
    z_scores_aapl = (log_returns_aapl - aapl_stats['mean']) / aapl_stats['std']
    z_scores_sp500 = (log_returns_sp500 - sp500_stats['mean']) / sp500_stats['std']
    
    # Create figure with single subplot for standardized log returns
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot overlapping histogram of standardized log returns (z-scores)
    z_min, z_max = -3, 4
    bins_z = np.linspace(z_min, z_max, 25)
    
    # Plot standardized log returns histograms
    ax.hist(z_scores_aapl, bins=bins_z, alpha=0.7, edgecolor='black', 
            color='lightblue', density=True, label='Apple (AAPL) Log Z-Scores')
    ax.hist(z_scores_sp500, bins=bins_z, alpha=0.7, edgecolor='black', 
            color='lightcoral', density=True, label='S&P 500 (^GSPC) Log Z-Scores')
    
    # Add standard normal distribution overlay
    z_range = np.linspace(z_min, z_max, 200)
    standard_normal = stats.norm.pdf(z_range, 0, 1)
    ax.plot(z_range, standard_normal, 'k-', linewidth=3, label='Standard Normal Distribution')
    
    # Add reference lines at z = -1, 0, +1
    ax.axvline(-1, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Z = -1 (1 SD below mean)')
    ax.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Z = 0 (mean)')
    ax.axvline(1, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Z = +1 (1 SD above mean)')
    
    ax.set_title('Standardized Log Returns Comparison: AAPL vs S&P 500', fontsize=16, fontweight='bold')
    ax.set_xlabel('Z-Score (Standardized Log Return)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_xlim(z_min, z_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    
    # Add summary box
    aapl_sharpe = aapl_stats['mean'] / aapl_stats['std']
    sp500_sharpe = sp500_stats['mean'] / sp500_stats['std']
    
    summary_text = f"""Log Returns Standardization Summary:

Raw Log Statistics:
• AAPL Log Mean: {aapl_stats['mean']:.1f}% | Log Volatility: {aapl_stats['std']:.1f}%
• S&P 500 Log Mean: {sp500_stats['mean']:.1f}% | Log Volatility: {sp500_stats['std']:.1f}%

Sharpe Ratios (Log Mean/Log Volatility):
• AAPL Log Sharpe: {aapl_sharpe:.3f}
• S&P 500 Log Sharpe: {sp500_sharpe:.3f}

Z-Score Interpretation:
• Z = 0: Log return equals the mean
• Z = +1: Log return is 1 standard deviation above mean
• Z = -1: Log return is 1 standard deviation below mean
• Z = +2: Log return is 2 standard deviations above mean

Log Returns Benefits:
• More symmetric distribution than raw returns
• Better for statistical modeling
• Time-additive property preserved
• Outliers less extreme"""
    
    fig.text(0.75, 0.02, summary_text, fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'apple_vs_sp500_log_returns_standardized.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print z-score statistics
    print(f"\n📊 Log Returns Z-Score Statistics:")
    print(f"  AAPL Log Z-Score Range: {z_scores_aapl.min():.2f} to {z_scores_aapl.max():.2f}")
    print(f"  S&P 500 Log Z-Score Range: {z_scores_sp500.min():.2f} to {z_scores_sp500.max():.2f}")
    print(f"  AAPL Log Z-Score Mean: {z_scores_aapl.mean():.3f} (should be ~0)")
    print(f"  S&P 500 Log Z-Score Mean: {z_scores_sp500.mean():.3f} (should be ~0)")
    print(f"  AAPL Log Z-Score Std Dev: {z_scores_aapl.std():.3f} (should be ~1)")
    print(f"  S&P 500 Log Z-Score Std Dev: {z_scores_sp500.std():.3f} (should be ~1)")

def create_apple_vs_stock_log_comparison(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker):
    """Create comprehensive Apple vs individual stock comparison using log returns (similar to Apple vs S&P 500)."""
    print(f"📊 Creating Apple vs {stock_ticker} log returns detailed comparison...")
    
    log_returns_aapl = aapl_data['Log_Return']
    log_returns_stock = stock_data['Log_Return']
    
    # Create main comparison visualization (4-panel layout)
    create_apple_vs_stock_log_main_comparison(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker)
    
    # Create focused comparison plot (2-panel layout)
    create_apple_vs_stock_log_focused_comparison(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker)
    
    # Create standardized log returns analysis
    create_apple_vs_stock_log_standardized_analysis(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker)
    
    # Print detailed comparison analysis
    print_apple_vs_stock_log_analysis(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker)

def create_apple_vs_stock_log_main_comparison(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker):
    """Create the main 4-panel comparison visualization for Apple vs individual stock using log returns."""
    print(f"📊 Creating main log returns comparison visualization for Apple vs {stock_ticker}...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    log_returns_aapl = aapl_data['Log_Return']
    log_returns_stock = stock_data['Log_Return']
    
    # Plot 1: Log returns comparison
    x_min, x_max = -80, 100
    bins_aapl = np.linspace(x_min, x_max, 20)
    bins_stock = np.linspace(x_min, x_max, 15)
    
    ax1.hist(log_returns_aapl, bins=bins_aapl, alpha=0.7, edgecolor='black', 
             color='lightblue', density=True, label='Apple (AAPL) Log Returns')
    ax1.hist(log_returns_stock, bins=bins_stock, alpha=0.7, edgecolor='black', 
             color='lightcoral', density=True, label=f'{stock_ticker} Log Returns')
    
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
    
    ax1.set_title('Log Returns Distribution Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Annual Log Return (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Log returns over time
    ax2.plot(aapl_data['Year'], log_returns_aapl, 'o-', linewidth=2, markersize=6, 
             label='Apple (AAPL)', color='blue')
    ax2.plot(stock_data['Year'], log_returns_stock, 's-', linewidth=2, markersize=6, 
             label=f'{stock_ticker}', color='red')
    
    ax2.axhline(y=aapl_stats['mean'], color='blue', linestyle='--', alpha=0.7, 
                label=f'AAPL Mean: {aapl_stats["mean"]:.1f}%')
    ax2.axhline(y=stock_stats['mean'], color='red', linestyle='--', alpha=0.7, 
                label=f'{stock_ticker} Mean: {stock_stats["mean"]:.1f}%')
    
    ax2.set_title('Annual Log Returns Over Time', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Annual Log Return (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Rolling Correlation
    combined_data = pd.merge(aapl_data, stock_data, on='Year', suffixes=('_AAPL', f'_{stock_ticker}'))
    if len(combined_data) >= 5:
        rolling_corr = combined_data['Log_Return_AAPL'].rolling(window=5, min_periods=1).corr(combined_data[f'Log_Return_{stock_ticker}'])
        
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
    📊 LOG RETURNS COMPARISON SUMMARY
    
    📈 Apple (AAPL) Log Returns:
    • Mean Annual Log Return: {aapl_stats['mean']:.2f}%
    • Log Return Volatility: {aapl_stats['std']:.2f}%
    • Median Log Return: {aapl_stats['median']:.2f}%
    • 68% Range: {aapl_stats['confidence_68_lower']:.1f}% to {aapl_stats['confidence_68_upper']:.1f}%
    
    📊 {stock_ticker} Log Returns:
    • Mean Annual Log Return: {stock_stats['mean']:.2f}%
    • Log Return Volatility: {stock_stats['std']:.2f}%
    • Median Log Return: {stock_stats['median']:.2f}%
    • 68% Range: {stock_stats['confidence_68_lower']:.1f}% to {stock_stats['confidence_68_upper']:.1f}%
    
    🎯 Key Insights (Log Returns):
    • Apple Log Outperformance: {aapl_stats['mean'] - stock_stats['mean']:.1f}% per year
    • Apple Log Volatility: {aapl_stats['std'] - stock_stats['std']:.1f}% higher
    • Risk-Adjusted Log Return: {(aapl_stats['mean']/aapl_stats['std']) - (stock_stats['mean']/stock_stats['std']):.2f} higher
    
    📅 Data Coverage:
    • Apple: {len(aapl_data)} years ({aapl_data['Year'].min()}-{aapl_data['Year'].max()})
    • {stock_ticker}: {len(stock_data)} years ({stock_data['Year'].min()}-{stock_data['Year'].max()})
    
    💡 Log Returns Benefits:
    • More symmetric distribution
    • Better for statistical analysis
    • Time-additive property
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f'apple_vs_{stock_ticker.lower()}_log_returns_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_apple_vs_stock_log_focused_comparison(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker):
    """Create focused 2-panel comparison for Apple vs individual stock using log returns."""
    print(f"📊 Creating focused log returns comparison for Apple vs {stock_ticker}...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    log_returns_aapl = aapl_data['Log_Return']
    log_returns_stock = stock_data['Log_Return']
    
    # Plot 1: Log returns distribution comparison
    x_min, x_max = -80, 100
    bins_aapl = np.linspace(x_min, x_max, 20)
    bins_stock = np.linspace(x_min, x_max, 15)
    
    ax1.hist(log_returns_aapl, bins=bins_aapl, alpha=0.7, edgecolor='black', 
             color='lightblue', density=True, label='Apple (AAPL) Log Returns')
    ax1.hist(log_returns_stock, bins=bins_stock, alpha=0.7, edgecolor='black', 
             color='lightcoral', density=True, label=f'{stock_ticker} Log Returns')
    
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
    
    ax1.set_title('Log Returns Distribution Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Annual Log Return (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Log returns over time
    ax2.plot(aapl_data['Year'], log_returns_aapl, 'o-', linewidth=2, markersize=6, 
             label='Apple (AAPL)', color='blue')
    ax2.plot(stock_data['Year'], log_returns_stock, 's-', linewidth=2, markersize=6, 
             label=f'{stock_ticker}', color='red')
    
    # Add mean lines
    ax2.axhline(y=aapl_stats['mean'], color='blue', linestyle='--', alpha=0.7, 
                label=f'AAPL Mean: {aapl_stats["mean"]:.1f}%')
    ax2.axhline(y=stock_stats['mean'], color='red', linestyle='--', alpha=0.7, 
                label=f'{stock_ticker} Mean: {stock_stats["mean"]:.1f}%')
    
    ax2.set_title('Annual Log Returns Over Time', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Annual Log Return (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f'apple_vs_{stock_ticker.lower()}_log_returns_focused.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_apple_vs_stock_log_standardized_analysis(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker):
    """Create standardized log returns comparison analysis for Apple vs individual stock."""
    print(f"📊 Creating standardized log returns analysis for Apple vs {stock_ticker}...")
    
    log_returns_aapl = aapl_data['Log_Return']
    log_returns_stock = stock_data['Log_Return']
    
    # Calculate z-scores (standardized log returns)
    z_scores_aapl = (log_returns_aapl - aapl_stats['mean']) / aapl_stats['std']
    z_scores_stock = (log_returns_stock - stock_stats['mean']) / stock_stats['std']
    
    # Create figure with single subplot for standardized log returns only
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot overlapping histogram of standardized log returns (z-scores)
    z_min, z_max = -3, 4
    bins_z = np.linspace(z_min, z_max, 25)
    
    # Plot standardized log returns histograms
    ax.hist(z_scores_aapl, bins=bins_z, alpha=0.7, edgecolor='black', 
            color='lightblue', density=True, label='Apple (AAPL) Log Z-Scores')
    ax.hist(z_scores_stock, bins=bins_z, alpha=0.7, edgecolor='black', 
            color='lightcoral', density=True, label=f'{stock_ticker} Log Z-Scores')
    
    # Add standard normal distribution overlay
    z_range = np.linspace(z_min, z_max, 200)
    standard_normal = stats.norm.pdf(z_range, 0, 1)
    ax.plot(z_range, standard_normal, 'k-', linewidth=3, label='Standard Normal Distribution')
    
    # Add reference lines at z = -1, 0, +1
    ax.axvline(-1, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Z = -1 (1 SD below mean)')
    ax.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Z = 0 (mean)')
    ax.axvline(1, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Z = +1 (1 SD above mean)')
    
    ax.set_title(f'Standardized Log Returns Comparison: AAPL vs {stock_ticker}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Z-Score (Standardized Log Return)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_xlim(z_min, z_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    
    # Add summary box
    aapl_sharpe = aapl_stats['mean'] / aapl_stats['std']
    stock_sharpe = stock_stats['mean'] / stock_stats['std']
    
    summary_text = f"""Log Returns Standardization Summary:

Raw Log Statistics:
• AAPL Log Mean: {aapl_stats['mean']:.1f}% | Log Volatility: {aapl_stats['std']:.1f}%
• {stock_ticker} Log Mean: {stock_stats['mean']:.1f}% | Log Volatility: {stock_stats['std']:.1f}%

Sharpe Ratios (Log Mean/Log Volatility):
• AAPL Log Sharpe: {aapl_sharpe:.3f}
• {stock_ticker} Log Sharpe: {stock_sharpe:.3f}

Z-Score Interpretation:
• Z = 0: Log return equals the mean
• Z = +1: Log return is 1 standard deviation above mean
• Z = -1: Log return is 1 standard deviation below mean
• Z = +2: Log return is 2 standard deviations above mean

Log Returns Benefits:
• More symmetric distribution than raw returns
• Better for statistical modeling
• Time-additive property preserved
• Outliers less extreme"""
    
    fig.text(0.75, 0.02, summary_text, fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f'apple_vs_{stock_ticker.lower()}_log_returns_standardized.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print z-score statistics
    print(f"\n📊 Log Returns Z-Score Statistics for Apple vs {stock_ticker}:")
    print(f"  AAPL Log Z-Score Range: {z_scores_aapl.min():.2f} to {z_scores_aapl.max():.2f}")
    print(f"  {stock_ticker} Log Z-Score Range: {z_scores_stock.min():.2f} to {z_scores_stock.max():.2f}")
    print(f"  AAPL Log Z-Score Mean: {z_scores_aapl.mean():.3f} (should be ~0)")
    print(f"  {stock_ticker} Log Z-Score Mean: {z_scores_stock.mean():.3f} (should be ~0)")
    print(f"  AAPL Log Z-Score Std Dev: {z_scores_aapl.std():.3f} (should be ~1)")
    print(f"  {stock_ticker} Log Z-Score Std Dev: {z_scores_stock.std():.3f} (should be ~1)")

def print_apple_vs_stock_log_analysis(aapl_data, stock_data, aapl_stats, stock_stats, stock_ticker):
    """Print detailed comparison analysis for Apple vs individual stock using log returns."""
    print(f"\n📊 APPLE VS {stock_ticker} LOG RETURNS COMPARISON ANALYSIS")
    print("=" * 60)
    
    print(f"\n📈 Log Returns Performance Comparison:")
    print(f"  Apple Mean Annual Log Return: {aapl_stats['mean']:.2f}%")
    print(f"  {stock_ticker} Mean Annual Log Return: {stock_stats['mean']:.2f}%")
    print(f"  Apple Log Outperformance: {aapl_stats['mean'] - stock_stats['mean']:.2f}% per year")
    
    print(f"\n📊 Log Returns Risk Comparison:")
    print(f"  Apple Log Volatility: {aapl_stats['std']:.2f}%")
    print(f"  {stock_ticker} Log Volatility: {stock_stats['std']:.2f}%")
    print(f"  Log Volatility Difference: {aapl_stats['std'] - stock_stats['std']:.2f}%")
    
    print(f"\n🎯 Risk-Adjusted Log Returns:")
    aapl_sharpe = aapl_stats['mean'] / aapl_stats['std']
    stock_sharpe = stock_stats['mean'] / stock_stats['std']
    print(f"  Apple Log Sharpe Ratio: {aapl_sharpe:.3f}")
    print(f"  {stock_ticker} Log Sharpe Ratio: {stock_sharpe:.3f}")
    print(f"  Log Sharpe Ratio Difference: {aapl_sharpe - stock_sharpe:.3f}")
    
    # Calculate correlation
    combined_data = pd.merge(aapl_data, stock_data, on='Year', suffixes=('_AAPL', f'_{stock_ticker}'))
    correlation = combined_data['Log_Return_AAPL'].corr(combined_data[f'Log_Return_{stock_ticker}'])
    print(f"\n🔗 Log Returns Correlation Analysis:")
    print(f"  Correlation between Apple and {stock_ticker} log returns: {correlation:.3f}")
    
    print(f"\n📅 Log Returns Performance by Year:")
    print("Year   Apple    {}  Difference".format(stock_ticker))
    print("-" * 50)
    
    for _, row in combined_data.iterrows():
        apple_log_return = row['Log_Return_AAPL']
        stock_log_return = row[f'Log_Return_{stock_ticker}']
        difference = apple_log_return - stock_log_return
        
        # Add emoji indicators
        apple_emoji = "🟢" if apple_log_return > 0 else "🔴"
        stock_emoji = "🟢" if stock_log_return > 0 else "🔴"
        
        print(f"{row['Year']:.1f}  {apple_log_return:6.1f}% {apple_emoji} {stock_log_return:6.1f}% {stock_emoji} {difference:8.1f}%")
    
    print(f"\n🏆 Best Years (Log Returns):")
    print(f"  Apple: {aapl_stats['best_year']['Year']:.0f} ({aapl_stats['best_year']['Log_Return']:.1f}%)")
    print(f"  {stock_ticker}: {stock_stats['best_year']['Year']:.0f} ({stock_stats['best_year']['Log_Return']:.1f}%)")
    
    print(f"\n📉 Worst Years (Log Returns):")
    print(f"  Apple: {aapl_stats['worst_year']['Year']:.0f} ({aapl_stats['worst_year']['Log_Return']:.1f}%)")
    print(f"  {stock_ticker}: {stock_stats['worst_year']['Year']:.0f} ({stock_stats['worst_year']['Log_Return']:.1f}%)")

def print_log_returns_comparison_analysis(aapl_data, sp500_data, aapl_stats, sp500_stats):
    """Print detailed comparison analysis for log returns."""
    print("\n📊 APPLE VS S&P 500 LOG RETURNS COMPARISON ANALYSIS")
    print("=" * 60)
    
    print(f"\n📈 Log Returns Performance Comparison:")
    print(f"  Apple Mean Annual Log Return: {aapl_stats['mean']:.2f}%")
    print(f"  S&P 500 Mean Annual Log Return: {sp500_stats['mean']:.2f}%")
    print(f"  Apple Log Outperformance: {aapl_stats['mean'] - sp500_stats['mean']:.2f}% per year")
    
    print(f"\n📊 Log Returns Risk Comparison:")
    print(f"  Apple Log Volatility: {aapl_stats['std']:.2f}%")
    print(f"  S&P 500 Log Volatility: {sp500_stats['std']:.2f}%")
    print(f"  Log Volatility Difference: {aapl_stats['std'] - sp500_stats['std']:.2f}%")
    
    print(f"\n🎯 Risk-Adjusted Log Returns:")
    aapl_sharpe = aapl_stats['mean'] / aapl_stats['std']
    sp500_sharpe = sp500_stats['mean'] / sp500_stats['std']
    print(f"  Apple Log Sharpe Ratio: {aapl_sharpe:.3f}")
    print(f"  S&P 500 Log Sharpe Ratio: {sp500_sharpe:.3f}")
    print(f"  Log Sharpe Ratio Difference: {aapl_sharpe - sp500_sharpe:.3f}")
    
    # Calculate correlation
    combined_data = pd.merge(aapl_data, sp500_data, on='Year', suffixes=('_AAPL', '_SP500'))
    correlation = combined_data['Log_Return_AAPL'].corr(combined_data['Log_Return_SP500'])
    print(f"\n🔗 Log Returns Correlation Analysis:")
    print(f"  Correlation between Apple and S&P 500 log returns: {correlation:.3f}")
    
    print(f"\n📅 Log Returns Performance by Year:")
    print("Year   Apple    S&P 500  Difference")
    print("-" * 40)
    
    for _, row in combined_data.iterrows():
        apple_log_return = row['Log_Return_AAPL']
        sp500_log_return = row['Log_Return_SP500']
        difference = apple_log_return - sp500_log_return
        
        # Add emoji indicators
        apple_emoji = "🟢" if apple_log_return > 0 else "🔴"
        sp500_emoji = "🟢" if sp500_log_return > 0 else "🔴"
        
        print(f"{row['Year']:.1f}  {apple_log_return:6.1f}% {apple_emoji} {sp500_log_return:6.1f}% {sp500_emoji} {difference:8.1f}%")
    
    print(f"\n🏆 Best Years (Log Returns):")
    print(f"  Apple: {aapl_stats['best_year']['Year']:.0f} ({aapl_stats['best_year']['Log_Return']:.1f}%)")
    print(f"  S&P 500: {sp500_stats['best_year']['Year']:.0f} ({sp500_stats['best_year']['Log_Return']:.1f}%)")
    
    print(f"\n📉 Worst Years (Log Returns):")
    print(f"  Apple: {aapl_stats['worst_year']['Year']:.0f} ({aapl_stats['worst_year']['Log_Return']:.1f}%)")
    print(f"  S&P 500: {sp500_stats['worst_year']['Year']:.0f} ({sp500_stats['worst_year']['Log_Return']:.1f}%)")

# ============================================================================
# ENHANCED ROLLING CORRELATION ANALYSIS (LOG RETURNS)
# ============================================================================

def create_enhanced_rolling_correlation_log_analysis():
    """Create enhanced rolling correlation analysis with multiple time windows and macro events using log returns for Apple vs high correlation stocks."""
    print("📊 Creating enhanced rolling correlation log returns analysis for Apple vs high correlation stocks...")
    
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
    print("\n🍎 Creating Apple vs S&P 500 rolling correlation log returns analysis...")
    create_rolling_correlation_log_analysis(aapl_data, sp500_data, 'AAPL', 'S&P 500')
    
    # Create rolling correlation analysis for Apple vs high correlation stocks
    for ticker, data in high_corr_stocks.items():
        print(f"\n🍎 Creating Apple vs {ticker} rolling correlation log returns analysis...")
        create_rolling_correlation_log_analysis(aapl_data, data, 'AAPL', ticker)

def create_rolling_correlation_log_analysis(stock_data, sp500_data, stock_ticker, benchmark_ticker):
    """Create comprehensive rolling correlation analysis with multiple time windows using log returns."""
    print(f"📊 Creating rolling correlation log returns analysis for {stock_ticker} vs {benchmark_ticker}...")
    
    # Get close prices
    stock_close_col = [col for col in stock_data.columns if 'Close' in col][0]
    sp500_close_col = [col for col in sp500_data.columns if 'Close' in col][0]
    
    stock_prices = stock_data[stock_close_col]
    sp500_prices = sp500_data[sp500_close_col]
    
    # Calculate daily log returns
    stock_log_returns = np.log(stock_prices / stock_prices.shift(1)) * 100
    sp500_log_returns = np.log(sp500_prices / sp500_prices.shift(1)) * 100
    
    # Align data by date
    combined_log_returns = pd.DataFrame({
        f'{stock_ticker}_Log_Return': stock_log_returns,
        f'{benchmark_ticker}_Log_Return': sp500_log_returns
    }).dropna()
    
    if len(combined_log_returns) < 252:  # Need at least 1 year of data
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
    fig.suptitle(f'{stock_ticker} vs {benchmark_ticker}: Rolling Log Returns Correlations Across Different Time Windows', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (window_name, window_days) in enumerate(windows.items()):
        ax = axes[i]
        
        # Calculate rolling correlation
        rolling_corr = combined_log_returns[f'{stock_ticker}_Log_Return'].rolling(
            window=window_days, min_periods=window_days//2
        ).corr(combined_log_returns[f'{benchmark_ticker}_Log_Return'])
        
        # Plot the correlation
        ax.plot(combined_log_returns.index, rolling_corr, 'b-', linewidth=2, 
                label=f'{window_name} Rolling Correlation')
        
        # Add reference lines
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, 
                  label='Strong Positive Correlation (0.7)')
        ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.7, 
                  label='No Correlation (0.0)')
        ax.axhline(y=-0.7, color='red', linestyle='--', alpha=0.7, 
                  label='Strong Negative Correlation (-0.7)')
        
        # Add major macro events
        add_macro_events_log(ax, combined_log_returns.index)
        
        # Formatting
        ax.set_title(f'{window_name} Rolling Correlation (Log Returns)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Correlation', fontsize=10)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f'{stock_ticker}_vs_{benchmark_ticker.replace(" ", "_")}_log_rolling_correlations.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print correlation statistics
    print(f"\n📊 {stock_ticker} vs {benchmark_ticker} Rolling Log Returns Correlation Statistics:")
    for window_name, window_days in windows.items():
        rolling_corr = combined_log_returns[f'{stock_ticker}_Log_Return'].rolling(
            window=window_days, min_periods=window_days//2
        ).corr(combined_log_returns[f'{benchmark_ticker}_Log_Return'])
        
        print(f"  {window_name}:")
        print(f"    Mean Correlation: {rolling_corr.mean():.3f}")
        print(f"    Std Deviation: {rolling_corr.std():.3f}")
        print(f"    Min Correlation: {rolling_corr.min():.3f}")
        print(f"    Max Correlation: {rolling_corr.max():.3f}")

def add_macro_events_log(ax, date_index):
    """Add major macro events as vertical lines on the log returns correlation plot."""
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

def create_mag7_log_comparison_visualization(mag7_data_dict, mag7_stats_dict):
    """Create comprehensive comparison visualization for all MAG7 stocks using log returns."""
    print("📊 Creating MAG7 log returns comparison visualization...")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Colors for MAG7 stocks
    colors = ['#007AFF', '#00A4EF', '#FF9900', '#4285F4', '#76B900', '#CC0000', '#1877F2']
    
    # Plot 1: Log returns comparison
    x_min, x_max = -80, 100
    bins = np.linspace(x_min, x_max, 30)
    
    for i, ticker in enumerate(MAG7_TICKERS):
        if ticker in mag7_data_dict:
            log_returns = mag7_data_dict[ticker]['Log_Return']
            ax1.hist(log_returns, bins=bins, alpha=0.7, edgecolor='black', 
                    color=colors[i], density=True, label=f'{ticker} Log Returns')
    
    # Add normal distribution overlays
    x_range = np.linspace(x_min, x_max, 200)
    for i, ticker in enumerate(MAG7_TICKERS):
        if ticker in mag7_stats_dict:
            stats_dict = mag7_stats_dict[ticker]
            normal = stats.norm.pdf(x_range, stats_dict['mean'], stats_dict['std'])
            ax1.plot(x_range, normal, color=colors[i], linewidth=2, 
                    linestyle='--', alpha=0.8, label=f'{ticker} Normal')
    
    ax1.set_title('MAG7 Stocks: Annual Log Returns Distribution', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Annual Log Return (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Mean log returns comparison
    tickers = list(mag7_stats_dict.keys())
    means = [mag7_stats_dict[ticker]['mean'] for ticker in tickers]
    stds = [mag7_stats_dict[ticker]['std'] for ticker in tickers]
    
    bars = ax2.bar(tickers, means, yerr=stds, capsize=5, alpha=0.7, color=colors[:len(tickers)])
    ax2.set_title('MAG7 Stocks: Mean Annual Log Returns with Volatility', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Annual Log Return (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Sharpe ratios (log returns)
    sharpe_ratios = [mag7_stats_dict[ticker]['mean'] / mag7_stats_dict[ticker]['std'] 
                    for ticker in tickers]
    
    bars = ax3.bar(tickers, sharpe_ratios, alpha=0.7, color=colors[:len(tickers)])
    ax3.set_title('MAG7 Stocks: Log Returns Sharpe Ratios (Mean/Volatility)', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Sharpe Ratio', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, sharpe in zip(bars, sharpe_ratios):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{sharpe:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Correlation heatmap
    # Calculate correlation matrix for all stocks
    all_log_returns = pd.DataFrame()
    for ticker in MAG7_TICKERS:
        if ticker in mag7_data_dict:
            all_log_returns[ticker] = mag7_data_dict[ticker]['Log_Return']
    
    if not all_log_returns.empty:
        correlation_matrix = all_log_returns.corr()
        
        im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_title('MAG7 Stocks: Annual Log Returns Correlation Matrix', fontweight='bold', fontsize=14)
        
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
    plt.savefig(os.path.join(GRAPHS_DIR, 'mag7_log_returns_comprehensive_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_individual_stock_log_analysis(ticker, annual_data, stats_dict):
    """Create individual analysis for each MAG7 stock using log returns."""
    print(f"📊 Creating individual log returns analysis for {ticker}...")
    
    log_returns = annual_data['Log_Return']
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Log returns over time
    ax1.plot(annual_data['Year'], log_returns, 'o-', linewidth=2, markersize=6)
    ax1.axhline(y=stats_dict['mean'], color='red', linestyle='--', 
                label=f'Mean: {stats_dict["mean"]:.1f}%')
    ax1.fill_between(annual_data['Year'], 
                     stats_dict['confidence_68_lower'], 
                     stats_dict['confidence_68_upper'], 
                     alpha=0.3, color='red', label='±1 Std Dev')
    
    ax1.set_title(f'{ticker}: Annual Log Returns Over Time', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Annual Log Return (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Log returns distribution
    x_min, x_max = log_returns.min() - 10, log_returns.max() + 10
    bins = np.linspace(x_min, x_max, 20)
    
    ax2.hist(log_returns, bins=bins, alpha=0.7, edgecolor='black', density=True)
    
    # Add normal distribution overlay
    x_range = np.linspace(x_min, x_max, 200)
    normal = stats.norm.pdf(x_range, stats_dict['mean'], stats_dict['std'])
    ax2.plot(x_range, normal, 'r-', linewidth=2, label='Normal Distribution')
    
    ax2.axvline(stats_dict['mean'], color='red', linestyle='--', 
                label=f'Mean: {stats_dict["mean"]:.1f}%')
    
    ax2.set_title(f'{ticker}: Log Returns Distribution', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Annual Log Return (%)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot
    ax3.boxplot(log_returns, patch_artist=True)
    ax3.set_title(f'{ticker}: Log Returns Box Plot', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Annual Log Return (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistical summary
    ax4.axis('off')
    
    summary_text = f"""
    📊 {ticker} LOG RETURNS STATISTICAL SUMMARY
    
    📈 Log Returns Performance:
    • Mean Annual Log Return: {stats_dict['mean']:.2f}%
    • Median Log Return: {stats_dict['median']:.2f}%
    • Log Volatility (Std Dev): {stats_dict['std']:.2f}%
    • Log Sharpe Ratio: {stats_dict['mean']/stats_dict['std']:.3f}
    
    📊 Log Returns Distribution:
    • Skewness: {stats_dict['skewness']:.3f}
    • Kurtosis: {stats_dict['kurtosis']:.3f}
    • 68% Range: {stats_dict['confidence_68_lower']:.1f}% to {stats_dict['confidence_68_upper']:.1f}%
    
    🏆 Best Year: {stats_dict['best_year']['Year']:.0f} ({stats_dict['best_year']['Log_Return']:.1f}%)
    📉 Worst Year: {stats_dict['worst_year']['Year']:.0f} ({stats_dict['worst_year']['Log_Return']:.1f}%)
    
    📅 Data Coverage: {stats_dict['total_years']} years
    
    💡 Log Returns Benefits:
    • More symmetric distribution
    • Better for statistical modeling
    • Time-additive property
    • Reduced fat tails
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f'{ticker}_log_returns_individual_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function for FMP log returns analysis with 35+ years of historical data."""
    print("🚀 FMP Log Returns Analysis Module (35+ Years of Data)")
    print("=" * 70)
    print("📊 Analyzing comprehensive historical data using log transformations")
    print("📅 Date range: 1990-2025 (35+ years)")
    print("🎯 MAG7 stocks + S&P 500 log returns analysis")
    print("📈 Enhanced statistical properties with log transformations")
    print("=" * 70)
    
    # Run Apple vs S&P 500 log returns analysis
    print("\n🍎 Running Apple vs S&P 500 log returns analysis (35+ years)...")
    apple_vs_sp500_results = create_apple_vs_sp500_log_analysis()
    
    # Run MAG7 log returns analysis
    print("\n🚀 Running MAG7 stocks log returns analysis (complete historical coverage)...")
    mag7_results = create_mag7_log_analysis()
    
    # Run enhanced rolling correlation analysis
    print("\n📊 Running enhanced rolling correlation log returns analysis...")
    create_enhanced_rolling_correlation_log_analysis()
    
    print("\n✅ FMP Log Returns Analysis complete!")
    print(f"\n📊 Check the 'graphs/log_returns_analysis' directory for comprehensive visualizations")
    print(f"📈 Analysis covers multiple market cycles with enhanced statistical properties")
    print(f"🎯 Professional-grade log returns insights from FMP data")
    print(f"💡 Log transformations provide better modeling foundation for predictions")

if __name__ == "__main__":
    main()

