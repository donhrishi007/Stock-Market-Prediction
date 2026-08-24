"""
Step 6: Quarterly Fundamentals Processing and Analysis
====================================================

This module loads quarterly fundamentals for AAPL, MSFT, GOOGL, AMZN from local FMP exports,
engineers features (QoQ, YoY, TTM), and creates slide-ready visualizations.

Key Features:
1. Load quarterly fundamentals from Excel files
2. Engineer TTM, YoY, QoQ features
3. Align by fiscal quarter (2005 onward)
4. Create 6 slide-ready charts
5. Save tidy dataset for ML

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

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
FUNDAMENTAL_TYPES = ['income_statement', 'balance_sheet', 'cash_flow', 'key_metrics']

# Input paths
INPUT_BASE_DIR = Path('fmp_data/fundamentals')
PRICE_DATA_DIR = Path('fmp_data/prices')

# Output paths
OUTPUT_DIR = Path('artifacts/fundamentals')
GRAPHS_DIR = Path('graphs/fundamentals_q')

# Ticker colors for consistent plotting
TICKER_COLORS = {
    'AAPL': '#1d4ed8',    # Blue
    'MSFT': '#059669',    # Green
    'GOOGL': '#dc2626',   # Red
    'AMZN': '#7c3aed'     # Purple
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def setup_directories():
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created output directories")

def load_ticker_prices(ticker):
    """Load price data for a ticker from financial_data.db."""
    print(f"\n📈 Loading {ticker} price data from database...")
    
    db_path = Path('financial_data.db')
    if not db_path.exists():
        print(f"  ⚠️ Database not found: {db_path}")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Query price data for the ticker
        table_name = f"{ticker}_price"
        query = f"""
        SELECT Date as date, Open as open, High as high, Low as low, Close as close, Volume as volume
        FROM {table_name}
        ORDER BY Date
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print(f"  ⚠️ No price data found for {ticker}")
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"  ✅ Loaded price data: {len(df)} records from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        return df
        
    except Exception as e:
        print(f"  ❌ Error loading price data: {e}")
        return None

def load_ticker_fundamentals(ticker):
    """Load all fundamental data for a ticker."""
    print(f"\n📊 Loading {ticker} quarterly fundamentals...")
    
    ticker_data = {}
    # Fix: Apple folder is named APPL, not AAPL
    ticker_folder = 'APPL' if ticker == 'AAPL' else ticker
    ticker_dir = INPUT_BASE_DIR / ticker_folder / 'quarterly'
    
    # Map fundamental types to actual file names (using ticker symbols)
    # Special case for Apple (appl instead of aapl)
    file_prefix = 'appl' if ticker == 'AAPL' else ticker.lower()
    
    file_mapping = {
        'income_statement': f"{file_prefix}_income_statement.xlsx",  # Fix: .xls to .xlsx
        'balance_sheet': f"{file_prefix}_balance_sheet.xlsx",        # Fix: .xls to .xlsx
        'cash_flow': f"{file_prefix}_cash_flow.xlsx",                # Fix: .xls to .xlsx
        'key_metrics': f"{file_prefix}_key_metrics.xlsx"             # Fix: .xls to .xlsx
    }
    
    for fundamental_type in FUNDAMENTAL_TYPES:
        file_name = file_mapping[fundamental_type]
        file_path = ticker_dir / file_name
        
        if file_path.exists():
            try:
                df = pd.read_excel(file_path)
                ticker_data[fundamental_type] = df
                print(f"  ✅ Loaded {fundamental_type}: {len(df)} records")
            except Exception as e:
                print(f"  ❌ Error loading {fundamental_type}: {e}")
                ticker_data[fundamental_type] = None
        else:
            print(f"  ⚠️ File not found: {file_path}")
            ticker_data[fundamental_type] = None
    
    return ticker_data

def process_ticker_data(ticker, ticker_data):
    """Process and clean fundamental data for a ticker."""
    print(f"\n🔄 Processing {ticker} data...")
    
    # Initialize result DataFrame
    result_df = pd.DataFrame()
    
    # Process each fundamental type
    for fundamental_type, df in ticker_data.items():
        if df is None or df.empty:
            continue
            
        print(f"  📋 Processing {fundamental_type}...")
        
        # Convert periodEndDate to datetime
        if 'periodEndDate' in df.columns:
            df['periodEndDate'] = pd.to_datetime(df['periodEndDate'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['periodEndDate'] = df['date']
        else:
            print(f"    ⚠️ No date column found in {fundamental_type}")
            continue
        
        # Create fiscal year and quarter
        df['fiscal_year'] = df['periodEndDate'].dt.year
        df['fiscal_quarter'] = df['periodEndDate'].dt.quarter
        
        # Select relevant columns based on fundamental type
        if fundamental_type == 'income_statement':
            columns_to_keep = ['periodEndDate', 'fiscal_year', 'fiscal_quarter', 
                             'revenue (Millions)', 'grossProfit (Millions)', 'operatingIncome (Millions)', 'netIncome (Millions)', 'eps']
        elif fundamental_type == 'balance_sheet':
            columns_to_keep = ['periodEndDate', 'fiscal_year', 'fiscal_quarter',
                             'totalDebt (Millions)', 'cashAndShortTermInvestments (Millions)']
        elif fundamental_type == 'cash_flow':
            columns_to_keep = ['periodEndDate', 'fiscal_year', 'fiscal_quarter',
                             'freeCashFlow (Millions)', 'operatingCashFlow (Millions)', 'capitalExpenditure (Millions)']
        elif fundamental_type == 'key_metrics':
            columns_to_keep = ['periodEndDate', 'fiscal_year', 'fiscal_quarter',
                             'weightedAverageShsOut (Millions)']
        
        # Keep only available columns
        available_columns = [col for col in columns_to_keep if col in df.columns]
        df_subset = df[available_columns].copy()
        
        # Convert numeric columns
        numeric_columns = df_subset.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['fiscal_year', 'fiscal_quarter']:
                df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
        
        # Merge with result DataFrame
        if result_df.empty:
            result_df = df_subset
        else:
            result_df = pd.merge(result_df, df_subset, on=['periodEndDate', 'fiscal_year', 'fiscal_quarter'], how='outer')
    
    # Add ticker identifier
    result_df['source_ticker'] = ticker
    
    # Sort by date
    result_df = result_df.sort_values('periodEndDate').reset_index(drop=True)
    
    # Keep all historical data (1995-2025)
    # Removed date filter to include full historical coverage
    
    # Engineer features (TTM, YoY, QoQ, margins, etc.)
    result_df = engineer_features(result_df)
    
    print(f"  ✅ Processed {ticker}: {len(result_df)} records from {result_df['periodEndDate'].min().strftime('%Y-%m-%d')} to {result_df['periodEndDate'].max().strftime('%Y-%m-%d')}")
    
    return result_df

def engineer_features(df):
    """Engineer TTM, YoY, QoQ features."""
    print(f"\n🔧 Engineering features for {df['source_ticker'].iloc[0]}...")
    
    # Sort by date to ensure proper calculations
    df = df.sort_values('periodEndDate').reset_index(drop=True)
    
    # Multi-timeframe TTM calculations (4Q, 8Q, 12Q)
    ttm_columns = ['revenue (Millions)', 'netIncome (Millions)', 'freeCashFlow (Millions)', 'eps']
    ttm_windows = [4, 8, 12]  # 4Q, 8Q, 12Q rolling averages
    
    for col in ttm_columns:
        if col in df.columns:
            clean_col = col.replace(' (Millions)', '').replace(' ', '_')
            for window in ttm_windows:
                if window == 4:
                    df[f'{clean_col}_ttm'] = df[col].rolling(window=window, min_periods=1).sum()
                else:
                    df[f'{clean_col}_ttm_{window}q'] = df[col].rolling(window=window, min_periods=1).sum()
    
    # Margin calculations (TTM)
    if 'revenue_ttm' in df.columns and 'grossProfit (Millions)' in df.columns:
        df['gross_margin_ttm'] = (df['grossProfit (Millions)'].rolling(window=4, min_periods=1).sum() / 
                                 df['revenue_ttm'] * 100)
    
    if 'revenue_ttm' in df.columns and 'operatingIncome (Millions)' in df.columns:
        df['op_margin_ttm'] = (df['operatingIncome (Millions)'].rolling(window=4, min_periods=1).sum() / 
                              df['revenue_ttm'] * 100)
    
    if 'revenue_ttm' in df.columns and 'netIncome_ttm' in df.columns:
        df['net_margin_ttm'] = (df['netIncome_ttm'] / df['revenue_ttm'] * 100)
    
    # YoY (Year-over-Year) growth calculations
    yoy_columns = ['revenue (Millions)', 'eps', 'freeCashFlow (Millions)']
    for col in yoy_columns:
        if col in df.columns:
            clean_col = col.replace(' (Millions)', '').replace(' ', '_')
            df[f'{clean_col}_yoy'] = df[col] / df[col].shift(4) - 1
    
    # QoQ (Quarter-over-Quarter) growth
    if 'revenue (Millions)' in df.columns:
        df['revenue_qoq'] = df['revenue (Millions)'].pct_change(1)
    
    # Shares outstanding change (buyback proxy)
    if 'weightedAverageShsOut (Millions)' in df.columns:
        df['shares_chg_yoy'] = df['weightedAverageShsOut (Millions)'].pct_change(4)
    
    # Financial Health Ratios
    if 'totalDebt (Millions)' in df.columns and 'totalAssets (Millions)' in df.columns:
        df['debt_to_assets'] = df['totalDebt (Millions)'] / df['totalAssets (Millions)']
    
    if 'totalDebt (Millions)' in df.columns and 'totalStockholdersEquity (Millions)' in df.columns:
        df['debt_to_equity'] = df['totalDebt (Millions)'] / df['totalStockholdersEquity (Millions)']
    
    if 'cashAndShortTermInvestments (Millions)' in df.columns and 'totalDebt (Millions)' in df.columns:
        df['cash_to_debt'] = df['cashAndShortTermInvestments (Millions)'] / df['totalDebt (Millions)']
    
    if 'totalAssets (Millions)' in df.columns and 'totalStockholdersEquity (Millions)' in df.columns:
        df['equity_multiplier'] = df['totalAssets (Millions)'] / df['totalStockholdersEquity (Millions)']
    
    # P/E ratio calculation (requires market cap and shares outstanding)
    if 'eps_ttm' in df.columns and 'weightedAverageShsOut (Millions)' in df.columns:
        # Calculate market cap (this would need price data, placeholder for now)
        # For now, we'll add a placeholder that can be filled with actual price data
        df['pe_ttm'] = np.nan  # Will be calculated when price data is available
    
    # Add reported_date using actual SEC filing dates from FMP
    # FMP provides 'fillingDate' which is the actual SEC filing date
    if 'fillingDate' in df.columns:
        df['fillingDate'] = pd.to_datetime(df['fillingDate'], errors='coerce')
        # Use actual filing date from FMP
        df['reported_date'] = df['fillingDate']
        print(f"  📅 Using actual SEC filing dates from FMP")
    else:
        # Fallback: Use 45-day lag if no filing date available
        df['reported_date'] = df['periodEndDate'] + pd.Timedelta(days=45)
        print(f"  ⚠️ No fillingDate available, using 45-day reporting lag")
    
    print(f"  ✅ Engineered features: TTM, YoY, QoQ, margins, buyback proxy, P/E placeholder")
    
    return df

def calculate_pe_ratios(df, price_df):
    """Calculate P/E ratios using price data."""
    if price_df is None or price_df.empty:
        return df
    
    print(f"  📊 Calculating P/E ratios for {df['source_ticker'].iloc[0]}...")
    
    # Get quarterly price data (end of quarter)
    quarterly_prices = []
    quarterly_dates = []
    
    for _, row in df.iterrows():
        quarter_end = row['periodEndDate']
        # Find closest price data
        price_data = price_df[price_df['date'] <= quarter_end]
        if not price_data.empty:
            quarterly_prices.append(price_data['close'].iloc[-1])
            quarterly_dates.append(quarter_end)
        else:
            quarterly_prices.append(np.nan)
            quarterly_dates.append(quarter_end)
    
    # Calculate P/E ratios
    if 'eps_ttm' in df.columns and quarterly_prices:
        df['quarterly_price'] = quarterly_prices
        df['pe_ttm'] = df['quarterly_price'] / df['eps_ttm']
        
        # Calculate market cap
        if 'weightedAverageShsOut (Millions)' in df.columns:
            df['market_cap_billions'] = (df['quarterly_price'] * df['weightedAverageShsOut (Millions)']) / 1000
    
    return df

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    """Main function for quarterly fundamentals processing."""
    print("📊 Step 6: Quarterly Fundamentals Processing and Analysis")
    print("=" * 70)
    print("📈 Tickers: AAPL, MSFT, GOOGL, AMZN")
    print("🔧 Features: TTM, YoY, QoQ, Margins, Buybacks")
    print("📊 Outputs: CSV, Charts, Documentation")
    print("=" * 70)
    
    # Setup directories
    setup_directories()
    
    # Load and process all tickers
    all_data = []
    
    for ticker in TICKERS:
        # Load ticker data
        ticker_data = load_ticker_fundamentals(ticker)
        
        # Process ticker data
        processed_df = process_ticker_data(ticker, ticker_data)
        
        if not processed_df.empty:
            # Engineer features
            featured_df = engineer_features(processed_df)
            all_data.append(featured_df)
    
    if not all_data:
        print("❌ No data processed. Exiting.")
        return
    
    # Concatenate all tickers
    print(f"\n🔄 Concatenating all ticker data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by ticker and date
    combined_df = combined_df.sort_values(['source_ticker', 'periodEndDate']).reset_index(drop=True)
    
    print(f"✅ Combined dataset: {len(combined_df)} records")
    print(f"📅 Date range: {combined_df['periodEndDate'].min().strftime('%Y-%m-%d')} to {combined_df['periodEndDate'].max().strftime('%Y-%m-%d')}")
    
    # Save outputs
    save_outputs(combined_df)
    
    # Create visualizations
    create_visualizations(combined_df)
    
    # Load price data and calculate P/E ratios for all tickers
    print(f"\n📈 Loading price data and calculating P/E ratios...")
    
    for ticker in TICKERS:
        price_df = load_ticker_prices(ticker)
        if price_df is not None:
            ticker_data = combined_df[combined_df['source_ticker'] == ticker].copy()
            if not ticker_data.empty:
                ticker_data = calculate_pe_ratios(ticker_data, price_df)
                # Update the combined dataframe with P/E data
                combined_df.loc[combined_df['source_ticker'] == ticker, 'pe_ttm'] = ticker_data['pe_ttm']
                combined_df.loc[combined_df['source_ticker'] == ticker, 'quarterly_price'] = ticker_data['quarterly_price']
                if 'market_cap_billions' in ticker_data.columns:
                    combined_df.loc[combined_df['source_ticker'] == ticker, 'market_cap_billions'] = ticker_data['market_cap_billions']
    
    # Get Apple price data for dashboard
    aapl_prices = load_ticker_prices('AAPL')
    
    # Create Apple-specific analysis
    create_apple_comprehensive_dashboard(combined_df, aapl_prices)
    create_apple_multi_timeframe_analysis(combined_df)
    create_apple_financial_health_ratios(combined_df)
    
    # Create peer comparison analysis
    create_peer_comparison_dashboard(combined_df)
    create_peer_growth_analysis(combined_df)
    create_peer_valuation_metrics(combined_df)
    
    print(f"\n✅ Step 6 Complete!")
    print(f"📁 Data saved to: {OUTPUT_DIR}")
    print(f"📊 Charts saved to: {GRAPHS_DIR}")
    print(f"🍎 Apple-specific analysis completed!")
    print(f"📊 Peer comparison analysis completed!")

def save_outputs(df):
    """Save processed data and documentation."""
    print(f"\n💾 Saving outputs...")
    
    # Save CSV
    csv_file = OUTPUT_DIR / 'fundamentals_quarterly_long.csv'
    df.to_csv(csv_file, index=False)
    print(f"✅ Saved CSV: {csv_file}")
    
    # Create column dictionary
    create_column_dictionary(df)

def create_column_dictionary(df):
    """Create column dictionary documentation."""
    md_content = """# Quarterly Fundamentals Column Dictionary

## Overview
This dataset contains quarterly fundamental data for AAPL, MSFT, GOOGL, and AMZN with engineered features for analysis and modeling.

## Core Columns
- `source_ticker`: Company ticker symbol
- `periodEndDate`: Fiscal period end date
- `fiscal_year`: Fiscal year
- `fiscal_quarter`: Fiscal quarter (1-4)
- `reported_date`: Approximate reporting date (periodEndDate + 30 days)

## Financial Metrics
- `revenue`: Quarterly revenue
- `grossProfit`: Quarterly gross profit
- `operatingIncome`: Quarterly operating income
- `netIncome`: Quarterly net income
- `eps`: Earnings per share
- `freeCashFlow`: Quarterly free cash flow
- `sharesOutstanding`: Shares outstanding
- `totalDebt`: Total debt
- `cashAndShortTermInvestments`: Cash and short-term investments

## Engineered Features

### TTM (Trailing Twelve Months)
- `revenue_ttm`: 4-quarter rolling sum of revenue
- `netIncome_ttm`: 4-quarter rolling sum of net income
- `freeCashFlow_ttm`: 4-quarter rolling sum of free cash flow
- `eps_ttm`: 4-quarter rolling sum of EPS
- `revenue_ttm_8q`: 8-quarter rolling sum of revenue
- `revenue_ttm_12q`: 12-quarter rolling sum of revenue
- `netIncome_ttm_8q`: 8-quarter rolling sum of net income
- `netIncome_ttm_12q`: 12-quarter rolling sum of net income
- `freeCashFlow_ttm_8q`: 8-quarter rolling sum of free cash flow
- `freeCashFlow_ttm_12q`: 12-quarter rolling sum of free cash flow
- `eps_ttm_8q`: 8-quarter rolling sum of EPS
- `eps_ttm_12q`: 12-quarter rolling sum of EPS

### Margins (TTM)
- `gross_margin_ttm`: Gross margin percentage (TTM)
- `op_margin_ttm`: Operating margin percentage (TTM)
- `net_margin_ttm`: Net margin percentage (TTM)

### Growth Rates
- `revenue_yoy`: Year-over-year revenue growth
- `eps_yoy`: Year-over-year EPS growth
- `freeCashFlow_yoy`: Year-over-year FCF growth
- `revenue_qoq`: Quarter-over-quarter revenue growth

### Share Buybacks
- `shares_chg_yoy`: Year-over-year change in shares outstanding (negative = buybacks)

### Valuation Metrics
- `pe_ttm`: Price-to-Earnings TTM ratio (placeholder for price data integration)

### Financial Health Ratios
- `debt_to_assets`: Total debt divided by total assets
- `debt_to_equity`: Total debt divided by total stockholders equity
- `cash_to_debt`: Cash and short-term investments divided by total debt
- `equity_multiplier`: Total assets divided by total stockholders equity (leverage measure)

## Usage Notes
- Data filtered from 2005 onward
- TTM calculations smooth seasonality
- YoY growth controls for seasonality
- reported_date prevents future data leakage in backtesting
- Apple fiscal year ends in September; other companies may differ

## Data Quality
- Missing values handled with forward-fill where appropriate
- All monetary values in original currency units
- Percentages expressed as decimals (0.1 = 10%)
"""
    
    md_file = OUTPUT_DIR / 'fundamentals_quarterly_dict.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"✅ Saved documentation: {md_file}")

def create_visualizations(df):
    """Create slide-ready visualizations."""
    print(f"\n📊 Creating visualizations...")
    
    # Set up matplotlib style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # Create all plots
    create_revenue_ttm_trend(df)
    create_eps_ttm_trend(df)
    create_net_margin_ttm_trend(df)
    create_fcf_vs_netincome_ttm(df)
    create_revenue_yoy_box_last5y(df)
    create_pe_trend(df)
    create_revenue_index_100(df)
    
    print(f"✅ Created 7 visualizations in {GRAPHS_DIR}")

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_revenue_ttm_trend(df):
    """Create revenue TTM trend chart."""
    plt.figure(figsize=(12, 8))
    
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if 'revenue_ttm' in ticker_data.columns and not ticker_data['revenue_ttm'].isna().all():
            # Convert to billions (data is already in millions)
            revenue_billions = ticker_data['revenue_ttm'] / 1000
            plt.plot(ticker_data['periodEndDate'], revenue_billions, 
                    label=ticker, color=TICKER_COLORS[ticker], linewidth=2)
    
    plt.title('Revenue TTM Trend (Billions USD)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Revenue TTM ($B)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add event bands
    add_event_bands(plt)
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'revenue_ttm_trend.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_eps_ttm_trend(df):
    """Create EPS TTM trend chart."""
    plt.figure(figsize=(12, 8))
    
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if 'eps_ttm' in ticker_data.columns and not ticker_data['eps_ttm'].isna().all():
            plt.plot(ticker_data['periodEndDate'], ticker_data['eps_ttm'], 
                    label=ticker, color=TICKER_COLORS[ticker], linewidth=2)
    
    plt.title('EPS TTM Trend', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('EPS TTM ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add event bands
    add_event_bands(plt)
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'eps_ttm_trend.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_net_margin_ttm_trend(df):
    """Create net margin TTM trend chart."""
    plt.figure(figsize=(12, 8))
    
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if 'net_margin_ttm' in ticker_data.columns and not ticker_data['net_margin_ttm'].isna().all():
            plt.plot(ticker_data['periodEndDate'], ticker_data['net_margin_ttm'], 
                    label=ticker, color=TICKER_COLORS[ticker], linewidth=2)
    
    plt.title('Net Margin TTM Trend', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Net Margin TTM (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(GRAPHS_DIR / 'net_margin_ttm_trend.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_fcf_vs_netincome_ttm(df):
    """Create FCF vs Net Income TTM chart (AAPL only)."""
    plt.figure(figsize=(12, 8))
    
    aapl_data = df[df['source_ticker'] == 'AAPL'].copy()
    
    if 'freeCashFlow_ttm' in aapl_data.columns and not aapl_data['freeCashFlow_ttm'].isna().all():
        fcf_billions = aapl_data['freeCashFlow_ttm'] / 1000  # Convert millions to billions
        plt.plot(aapl_data['periodEndDate'], fcf_billions, 
                label='Free Cash Flow TTM', color='#059669', linewidth=2)
    
    if 'netIncome_ttm' in aapl_data.columns and not aapl_data['netIncome_ttm'].isna().all():
        ni_billions = aapl_data['netIncome_ttm'] / 1000  # Convert millions to billions
        plt.plot(aapl_data['periodEndDate'], ni_billions, 
                label='Net Income TTM', color='#dc2626', linewidth=2)
    
    plt.title('Apple: Free Cash Flow vs Net Income TTM\nEarnings ↔ Cash Conversion', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Amount ($B)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(GRAPHS_DIR / 'fcf_vs_netincome_ttm.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_buybacks_shares_yoy(df):
    """Create shares outstanding YoY change chart."""
    plt.figure(figsize=(12, 8))
    
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if 'shares_chg_yoy' in ticker_data.columns and not ticker_data['shares_chg_yoy'].isna().all():
            # Convert to percentage
            shares_chg_pct = ticker_data['shares_chg_yoy'] * 100
            plt.plot(ticker_data['periodEndDate'], shares_chg_pct, 
                    label=ticker, color=TICKER_COLORS[ticker], linewidth=2)
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Shares Outstanding YoY Change\n(Negative = Buybacks)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('YoY Change (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(GRAPHS_DIR / 'buybacks_shares_yoy.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_revenue_yoy_box_last5y(df):
    """Create revenue YoY growth boxplot for last 5 years."""
    plt.figure(figsize=(12, 8))
    
    # Filter last 5 years
    last_5y_data = df[df['periodEndDate'] >= df['periodEndDate'].max() - pd.Timedelta(days=5*365)]
    
    if 'revenue_yoy' in last_5y_data.columns:
        # Convert to percentage
        last_5y_data['revenue_yoy_pct'] = last_5y_data['revenue_yoy'] * 100
        
        # Create boxplot data
        box_data = []
        labels = []
        colors = []
        
        for ticker in TICKERS:
            ticker_data = last_5y_data[last_5y_data['source_ticker'] == ticker]
            if not ticker_data['revenue_yoy_pct'].isna().all():
                box_data.append(ticker_data['revenue_yoy_pct'].dropna())
                labels.append(ticker)
                colors.append(TICKER_COLORS[ticker])
        
        if box_data:
            bp = plt.boxplot(box_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
    
    plt.title('Revenue YoY Growth Distribution (Last 5 Years)', fontsize=14, fontweight='bold')
    plt.xlabel('Company', fontsize=12)
    plt.ylabel('Revenue YoY Growth (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(GRAPHS_DIR / 'revenue_yoy_box_last5y.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_pe_trend(df):
    """Create P/E TTM trend chart."""
    plt.figure(figsize=(12, 8))
    
    has_data = False
    all_valid_pe = []
    
    # First pass: collect all valid P/E ratios to determine reasonable range
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if 'pe_ttm' in ticker_data.columns and not ticker_data['pe_ttm'].isna().all():
            valid_pe = ticker_data['pe_ttm'].replace([np.inf, -np.inf], np.nan)
            valid_pe = valid_pe[valid_pe > 0]  # Only positive values
            valid_pe = valid_pe[valid_pe < 200]  # Reasonable upper bound
            all_valid_pe.extend(valid_pe.dropna().tolist())
    
    # Calculate reasonable bounds (exclude extreme outliers)
    if all_valid_pe:
        all_valid_pe = np.array(all_valid_pe)
        q25, q75 = np.percentile(all_valid_pe, [25, 75])
        iqr = q75 - q25
        upper_bound = q75 + 2 * iqr  # More conservative than 1.5 * IQR
        lower_bound = max(0, q25 - 2 * iqr)
    else:
        upper_bound = 50  # Default reasonable upper bound
        lower_bound = 0
    
    # Second pass: plot with consistent bounds
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if 'pe_ttm' in ticker_data.columns and not ticker_data['pe_ttm'].isna().all():
            # Filter out extreme values using calculated bounds
            valid_pe = ticker_data['pe_ttm'].replace([np.inf, -np.inf], np.nan)
            valid_pe = valid_pe[valid_pe > lower_bound]
            valid_pe = valid_pe[valid_pe < upper_bound]
            
            if not valid_pe.empty and valid_pe.count() > 5:  # Need at least 5 data points
                valid_dates = ticker_data.loc[valid_pe.index, 'periodEndDate']
                plt.plot(valid_dates, valid_pe, 
                        label=ticker, color=TICKER_COLORS[ticker], linewidth=2)
                has_data = True
    
    if not has_data:
        # If no P/E data, show a message
        plt.text(0.5, 0.5, 'P/E TTM data not available\nfor all tickers', 
                ha='center', va='center', transform=plt.gca().transAxes, 
                fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        plt.title('P/E TTM Trend (Data Not Available)', fontsize=14, fontweight='bold')
    else:
        plt.title('P/E TTM Trend', fontsize=14, fontweight='bold')
        plt.legend()
        # Set reasonable y-axis limits
        plt.ylim(0, min(100, upper_bound * 1.1))  # Cap at 100 or 110% of upper bound
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('P/E TTM', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add event bands
    add_event_bands(plt)
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'pe_trend.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_revenue_index_100(df):
    """Create revenue indexed to 100 at first overlap across tickers."""
    plt.figure(figsize=(12, 8))
    
    # Find the first common date across all tickers
    common_dates = None
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker]
        if 'revenue_ttm' in ticker_data.columns and not ticker_data['revenue_ttm'].isna().all():
            ticker_dates = set(ticker_data['periodEndDate'].dt.date)
            if common_dates is None:
                common_dates = ticker_dates
            else:
                common_dates = common_dates.intersection(ticker_dates)
    
    if common_dates:
        first_common_date = min(common_dates)
        print(f"  📅 First common date for indexing: {first_common_date}")
        
        for ticker in TICKERS:
            ticker_data = df[df['source_ticker'] == ticker].copy()
            if 'revenue_ttm' in ticker_data.columns and not ticker_data['revenue_ttm'].isna().all():
                # Filter data from first common date onward
                ticker_data = ticker_data[ticker_data['periodEndDate'].dt.date >= first_common_date]
                
                if not ticker_data.empty:
                    # Get the first revenue value as base (100)
                    first_revenue = ticker_data['revenue_ttm'].iloc[0]
                    
                    # Calculate index (first value = 100)
                    revenue_index = (ticker_data['revenue_ttm'] / first_revenue) * 100
                    
                    plt.plot(ticker_data['periodEndDate'], revenue_index, 
                            label=ticker, color=TICKER_COLORS[ticker], linewidth=2)
    
    plt.title('Revenue Indexed to 100 (First Common Date)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Revenue Index (Base = 100)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=100, color='black', linestyle='--', alpha=0.5)
    
    # Add event bands
    add_event_bands(plt)
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'revenue_index_100.png', dpi=300, bbox_inches='tight')
    plt.close()

def add_event_bands(plt):
    """Add shaded event bands for major market events."""
    # 2008-09 Financial Crisis
    plt.axvspan(pd.Timestamp('2008-09-01'), pd.Timestamp('2009-03-31'), 
                alpha=0.2, color='red', label='2008-09 Financial Crisis')
    
    # 2020 COVID shock
    plt.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-06-30'), 
                alpha=0.2, color='orange', label='2020 COVID Shock')
    
    # 2022-23 Fed hiking cycle
    plt.axvspan(pd.Timestamp('2022-03-01'), pd.Timestamp('2023-07-31'), 
                alpha=0.2, color='purple', label='2022-23 Fed Hiking Cycle')

# ============================================================================
# APPLE-SPECIFIC ANALYSIS FUNCTIONS
# ============================================================================

def create_apple_comprehensive_dashboard(df, price_df=None):
    """Create comprehensive 2x2 dashboard for Apple analysis."""
    print(f"\n🍎 Creating Apple comprehensive dashboard...")
    
    # Filter Apple data
    aapl_data = df[df['source_ticker'] == 'AAPL'].copy()
    if aapl_data.empty:
        print("  ⚠️ No Apple data found")
        return
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Apple Inc. (AAPL) Comprehensive Financial Analysis', fontsize=16, fontweight='bold')
    
    # 1. Revenue vs Net Income Trends (Top-Left)
    if 'revenue_ttm' in aapl_data.columns and 'netIncome_ttm' in aapl_data.columns:
        revenue_billions = aapl_data['revenue_ttm'] / 1000
        netincome_billions = aapl_data['netIncome_ttm'] / 1000
        
        ax1.plot(aapl_data['periodEndDate'], revenue_billions, 
                label='Revenue TTM', color='#1d4ed8', linewidth=2)
        ax1.plot(aapl_data['periodEndDate'], netincome_billions, 
                label='Net Income TTM', color='#059669', linewidth=2)
        
        ax1.set_title('Revenue vs Net Income Trends', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Billions ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        add_event_bands(ax1)
    
    # 2. Operating Performance Trends (Top-Right)
    if 'operatingIncome (Millions)' in aapl_data.columns and 'operatingCashFlow (Millions)' in aapl_data.columns:
        op_income_billions = aapl_data['operatingIncome (Millions)'].rolling(window=4, min_periods=1).sum() / 1000
        op_cashflow_billions = aapl_data['operatingCashFlow (Millions)'].rolling(window=4, min_periods=1).sum() / 1000
        
        ax2.plot(aapl_data['periodEndDate'], op_income_billions, 
                label='Operating Income TTM', color='#f59e0b', linewidth=2)
        ax2.plot(aapl_data['periodEndDate'], op_cashflow_billions, 
                label='Operating Cash Flow TTM', color='#7c3aed', linewidth=2)
        
        ax2.set_title('Operating Performance Trends', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Billions ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        add_event_bands(ax2)
    
    # 3. Cash Position and Debt Analysis (Bottom-Left)
    if 'cashAndShortTermInvestments (Millions)' in aapl_data.columns and 'totalDebt (Millions)' in aapl_data.columns:
        cash_billions = aapl_data['cashAndShortTermInvestments (Millions)'] / 1000
        debt_billions = aapl_data['totalDebt (Millions)'] / 1000
        
        ax3.plot(aapl_data['periodEndDate'], cash_billions, 
                label='Cash & Short-term Investments', color='#059669', linewidth=2)
        ax3.plot(aapl_data['periodEndDate'], debt_billions, 
                label='Total Debt', color='#dc2626', linewidth=2)
        
        ax3.set_title('Cash Position vs Debt', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Billions ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        add_event_bands(ax3)
    
    # 4. Stock Price vs Cash Position (Bottom-Right)
    if price_df is not None and 'close' in price_df.columns:
        # Get quarterly price data (end of quarter)
        quarterly_prices = []
        quarterly_dates = []
        for _, row in aapl_data.iterrows():
            quarter_end = row['periodEndDate']
            # Find closest price data
            price_data = price_df[price_df['date'] <= quarter_end]
            if not price_data.empty:
                quarterly_prices.append(price_data['close'].iloc[-1])
                quarterly_dates.append(quarter_end)
        
        if quarterly_prices and 'cashAndShortTermInvestments (Millions)' in aapl_data.columns:
            cash_billions = aapl_data['cashAndShortTermInvestments (Millions)'] / 1000
            
            # Dual y-axis
            ax4_twin = ax4.twinx()
            
            ax4.plot(quarterly_dates, quarterly_prices, 
                    label='Stock Price', color='#dc2626', linewidth=2)
            ax4_twin.plot(aapl_data['periodEndDate'], cash_billions, 
                         label='Cash Position', color='#fbbf24', linewidth=2)
            
            ax4.set_title('Stock Price vs Cash Position', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Stock Price ($)', color='#dc2626')
            ax4_twin.set_ylabel('Cash (Billions $)', color='#fbbf24')
            ax4.legend(loc='upper left')
            ax4_twin.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            add_event_bands(ax4)
    
    # Format all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'apple_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Created Apple comprehensive dashboard")

def create_apple_multi_timeframe_analysis(df):
    """Create multi-timeframe rolling averages analysis for Apple."""
    print(f"\n📊 Creating Apple multi-timeframe analysis...")
    
    aapl_data = df[df['source_ticker'] == 'AAPL'].copy()
    if aapl_data.empty:
        print("  ⚠️ No Apple data found")
        return
    
    # Only use available metrics to avoid empty panels
    available_metrics = []
    metrics_to_check = [
        ('revenue (Millions)', 'Revenue (Billions)', 1000),
        ('netIncome (Millions)', 'Net Income (Billions)', 1000),
        ('operatingIncome (Millions)', 'Operating Income (Billions)', 1000),
        ('operatingCashFlow (Millions)', 'Operating Cash Flow (Billions)', 1000),
        ('cashAndShortTermInvestments (Millions)', 'Cash (Billions)', 1000),
        ('totalDebt (Millions)', 'Long Term Debt (Billions)', 1000),
        ('eps', 'EPS ($)', 1)
    ]
    
    # Check which metrics are available
    for col, title, divisor in metrics_to_check:
        if col in aapl_data.columns and not aapl_data[col].isna().all():
            available_metrics.append((col, title, divisor))
    
    # Create appropriate subplot layout based on available metrics
    num_metrics = len(available_metrics)
    if num_metrics == 0:
        print("  ⚠️ No available metrics for multi-timeframe analysis")
        return
    
    # Determine grid size
    if num_metrics <= 4:
        rows, cols = 2, 2
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    elif num_metrics <= 6:
        rows, cols = 2, 3
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        rows, cols = 3, 3
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    fig.suptitle('Apple Inc. (AAPL) Multi-Timeframe Analysis: Original vs Smoothed', 
                 fontsize=16, fontweight='bold')
    
    for i, (col, title, divisor) in enumerate(available_metrics):
        if i >= rows * cols:  # Don't exceed grid size
            break
            
        row, col_idx = i // cols, i % cols
        ax = axes[row, col_idx] if rows > 1 else axes[col_idx]
        
        # Original quarterly data
        original_data = aapl_data[col] / divisor
        ax.plot(aapl_data['periodEndDate'], original_data, 
               'o-', color='lightblue', alpha=0.7, linewidth=1, markersize=3,
               label='Original Quarterly')
        
        # Rolling averages
        for window, color, label in [(4, 'red', '4Q Rolling'), 
                                   (8, 'orange', '8Q Rolling'), 
                                   (12, 'green', '12Q Rolling')]:
            rolling_data = aapl_data[col].rolling(window=window, min_periods=1).sum() / divisor
            ax.plot(aapl_data['periodEndDate'], rolling_data, 
                   color=color, linewidth=2, label=label)
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(len(available_metrics), rows * cols):
        row, col_idx = i // cols, i % cols
        ax = axes[row, col_idx] if rows > 1 else axes[col_idx]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'apple_multi_timeframe_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Created Apple multi-timeframe analysis")

def create_apple_financial_health_ratios(df):
    """Create financial health ratios analysis for Apple."""
    print(f"\n💊 Creating Apple financial health ratios...")
    
    aapl_data = df[df['source_ticker'] == 'AAPL'].copy()
    if aapl_data.empty:
        print("  ⚠️ No Apple data found")
        return
    
    # Create 2x2 subplot layout for ratios
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Apple Inc. (AAPL) Financial Health Ratios', fontsize=16, fontweight='bold')
    
    # 1. Cash-to-Debt Ratio (Top-Left)
    if 'cash_to_debt' in aapl_data.columns and not aapl_data['cash_to_debt'].isna().all():
        ax1.plot(aapl_data['periodEndDate'], aapl_data['cash_to_debt'], 
                color='#059669', linewidth=2, label='Cash-to-Debt')
        ax1.set_title('Cash-to-Debt Ratio', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Ratio')
        ax1.grid(True, alpha=0.3)
        add_event_bands(ax1)
        ax1.legend()
    else:
        # Alternative: Cash position over time
        if 'cashAndShortTermInvestments (Millions)' in aapl_data.columns:
            cash_billions = aapl_data['cashAndShortTermInvestments (Millions)'] / 1000
            ax1.plot(aapl_data['periodEndDate'], cash_billions, 
                    color='#059669', linewidth=2, label='Cash Position')
            ax1.set_title('Cash Position Over Time', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Cash (Billions $)')
            ax1.grid(True, alpha=0.3)
            add_event_bands(ax1)
            ax1.legend()
    
    # 2. Net Margin TTM (Top-Right)
    if 'net_margin_ttm' in aapl_data.columns and not aapl_data['net_margin_ttm'].isna().all():
        ax2.plot(aapl_data['periodEndDate'], aapl_data['net_margin_ttm'], 
                color='#0891b2', linewidth=2, label='Net Margin TTM')
        ax2.set_title('Net Margin TTM', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Percentage (%)')
        ax2.grid(True, alpha=0.3)
        add_event_bands(ax2)
        ax2.legend()
    
    # 3. Revenue Growth (Bottom-Left)
    if 'revenue_yoy' in aapl_data.columns and not aapl_data['revenue_yoy'].isna().all():
        revenue_yoy_pct = aapl_data['revenue_yoy'] * 100
        ax3.plot(aapl_data['periodEndDate'], revenue_yoy_pct, 
                color='#dc2626', linewidth=2, label='Revenue YoY Growth')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Revenue YoY Growth', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Growth (%)')
        ax3.grid(True, alpha=0.3)
        add_event_bands(ax3)
        ax3.legend()
    else:
        # Alternative: Revenue TTM trend
        if 'revenue_ttm' in aapl_data.columns:
            revenue_billions = aapl_data['revenue_ttm'] / 1000
            ax3.plot(aapl_data['periodEndDate'], revenue_billions, 
                    color='#dc2626', linewidth=2, label='Revenue TTM')
            ax3.set_title('Revenue TTM Trend', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Revenue TTM (Billions $)')
            ax3.grid(True, alpha=0.3)
            add_event_bands(ax3)
            ax3.legend()
    
    # 4. Free Cash Flow TTM (Bottom-Right)
    if 'freeCashFlow_ttm' in aapl_data.columns and not aapl_data['freeCashFlow_ttm'].isna().all():
        fcf_billions = aapl_data['freeCashFlow_ttm'] / 1000
        ax4.plot(aapl_data['periodEndDate'], fcf_billions, 
                color='#7c3aed', linewidth=2, label='Free Cash Flow TTM')
        ax4.set_title('Free Cash Flow TTM', fontsize=12, fontweight='bold')
        ax4.set_ylabel('FCF TTM (Billions $)')
        ax4.grid(True, alpha=0.3)
        add_event_bands(ax4)
        ax4.legend()
    else:
        # Alternative: Operating Cash Flow
        if 'operatingCashFlow (Millions)' in aapl_data.columns:
            op_cf_billions = aapl_data['operatingCashFlow (Millions)'].rolling(window=4, min_periods=1).sum() / 1000
            ax4.plot(aapl_data['periodEndDate'], op_cf_billions, 
                    color='#7c3aed', linewidth=2, label='Operating Cash Flow TTM')
            ax4.set_title('Operating Cash Flow TTM', fontsize=12, fontweight='bold')
            ax4.set_ylabel('OCF TTM (Billions $)')
            ax4.grid(True, alpha=0.3)
            add_event_bands(ax4)
            ax4.legend()
    
    # Format all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'apple_financial_health_ratios.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Created Apple financial health ratios")

# ============================================================================
# PEER COMPARISON FUNCTIONS
# ============================================================================

def create_peer_comparison_dashboard(df):
    """Create comprehensive peer comparison dashboard."""
    print(f"\n📊 Creating peer comparison dashboard...")
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MAG7 Peer Comparison: AAPL vs MSFT vs GOOGL vs AMZN', fontsize=16, fontweight='bold')
    
    # 1. Revenue TTM Comparison (Top-Left)
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if 'revenue_ttm' in ticker_data.columns and not ticker_data['revenue_ttm'].isna().all():
            revenue_billions = ticker_data['revenue_ttm'] / 1000
            ax1.plot(ticker_data['periodEndDate'], revenue_billions, 
                    label=ticker, color=TICKER_COLORS[ticker], linewidth=2)
    
    ax1.set_title('Revenue TTM Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Revenue TTM ($B)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    add_event_bands(ax1)
    
    # 2. Net Margin TTM Comparison (Top-Right)
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if 'net_margin_ttm' in ticker_data.columns and not ticker_data['net_margin_ttm'].isna().all():
            ax2.plot(ticker_data['periodEndDate'], ticker_data['net_margin_ttm'], 
                    label=ticker, color=TICKER_COLORS[ticker], linewidth=2)
    
    ax2.set_title('Net Margin TTM Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Net Margin TTM (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    add_event_bands(ax2)
    
    # 3. P/E Ratio Comparison (Bottom-Left)
    all_valid_pe = []
    
    # First pass: collect all valid P/E ratios to determine reasonable range
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if 'pe_ttm' in ticker_data.columns and not ticker_data['pe_ttm'].isna().all():
            valid_pe = ticker_data['pe_ttm'].replace([np.inf, -np.inf], np.nan)
            valid_pe = valid_pe[valid_pe > 0]
            valid_pe = valid_pe[valid_pe < 200]
            all_valid_pe.extend(valid_pe.dropna().tolist())
    
    # Calculate reasonable bounds
    if all_valid_pe:
        all_valid_pe = np.array(all_valid_pe)
        q25, q75 = np.percentile(all_valid_pe, [25, 75])
        iqr = q75 - q25
        upper_bound = q75 + 2 * iqr
        lower_bound = max(0, q25 - 2 * iqr)
    else:
        upper_bound = 50
        lower_bound = 0
    
    # Second pass: plot with consistent bounds
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if 'pe_ttm' in ticker_data.columns and not ticker_data['pe_ttm'].isna().all():
            # Filter out extreme values using calculated bounds
            valid_pe = ticker_data['pe_ttm'].replace([np.inf, -np.inf], np.nan)
            valid_pe = valid_pe[valid_pe > lower_bound]
            valid_pe = valid_pe[valid_pe < upper_bound]
            
            if not valid_pe.empty and valid_pe.count() > 5:
                valid_dates = ticker_data.loc[valid_pe.index, 'periodEndDate']
                ax3.plot(valid_dates, valid_pe, 
                        label=ticker, color=TICKER_COLORS[ticker], linewidth=2)
    
    # Set reasonable y-axis limits
    ax3.set_ylim(0, min(100, upper_bound * 1.1))
    
    ax3.set_title('P/E TTM Comparison', fontsize=12, fontweight='bold')
    ax3.set_ylabel('P/E TTM')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    add_event_bands(ax3)
    
    # 4. Free Cash Flow TTM Comparison (Bottom-Right)
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if 'freeCashFlow_ttm' in ticker_data.columns and not ticker_data['freeCashFlow_ttm'].isna().all():
            fcf_billions = ticker_data['freeCashFlow_ttm'] / 1000
            ax4.plot(ticker_data['periodEndDate'], fcf_billions, 
                    label=ticker, color=TICKER_COLORS[ticker], linewidth=2)
    
    ax4.set_title('Free Cash Flow TTM Comparison', fontsize=12, fontweight='bold')
    ax4.set_ylabel('FCF TTM ($B)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    add_event_bands(ax4)
    
    # Format all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'peer_comparison_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Created peer comparison dashboard")

def create_peer_growth_analysis(df):
    """Create peer growth analysis visualization."""
    print(f"\n📈 Creating peer growth analysis...")
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MAG7 Growth Analysis: Revenue, Margins, and Efficiency', fontsize=16, fontweight='bold')
    
    # 1. Revenue YoY Growth (Top-Left)
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if 'revenue_yoy' in ticker_data.columns and not ticker_data['revenue_yoy'].isna().all():
            revenue_yoy_pct = ticker_data['revenue_yoy'] * 100
            ax1.plot(ticker_data['periodEndDate'], revenue_yoy_pct, 
                    label=ticker, color=TICKER_COLORS[ticker], linewidth=2)
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('Revenue YoY Growth', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Revenue YoY Growth (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    add_event_bands(ax1)
    
    # 2. Operating Margin TTM (Top-Right)
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if 'op_margin_ttm' in ticker_data.columns and not ticker_data['op_margin_ttm'].isna().all():
            ax2.plot(ticker_data['periodEndDate'], ticker_data['op_margin_ttm'], 
                    label=ticker, color=TICKER_COLORS[ticker], linewidth=2)
    
    ax2.set_title('Operating Margin TTM', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Operating Margin TTM (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    add_event_bands(ax2)
    
    # 3. Operating Cash Flow TTM (Bottom-Left)
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if 'operatingCashFlow (Millions)' in ticker_data.columns and not ticker_data['operatingCashFlow (Millions)'].isna().all():
            op_cf_billions = ticker_data['operatingCashFlow (Millions)'].rolling(window=4, min_periods=1).sum() / 1000
            ax3.plot(ticker_data['periodEndDate'], op_cf_billions, 
                    label=ticker, color=TICKER_COLORS[ticker], linewidth=2)
    
    ax3.set_title('Operating Cash Flow TTM', fontsize=12, fontweight='bold')
    ax3.set_ylabel('OCF TTM (Billions $)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    add_event_bands(ax3)
    
    # 4. Free Cash Flow TTM (Bottom-Right)
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if 'freeCashFlow_ttm' in ticker_data.columns and not ticker_data['freeCashFlow_ttm'].isna().all():
            fcf_billions = ticker_data['freeCashFlow_ttm'] / 1000
            ax4.plot(ticker_data['periodEndDate'], fcf_billions, 
                    label=ticker, color=TICKER_COLORS[ticker], linewidth=2)
    
    ax4.set_title('Free Cash Flow TTM', fontsize=12, fontweight='bold')
    ax4.set_ylabel('FCF TTM (Billions $)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    add_event_bands(ax4)
    
    # Format all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'peer_growth_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Created peer growth analysis")

def create_peer_valuation_metrics(df):
    """Create peer valuation metrics comparison."""
    print(f"\n💰 Creating peer valuation metrics...")
    
    # Get latest data for each ticker
    latest_data = []
    for ticker in TICKERS:
        ticker_data = df[df['source_ticker'] == ticker].copy()
        if not ticker_data.empty:
            latest = ticker_data.iloc[-1]  # Most recent quarter
            latest_data.append({
                'ticker': ticker,
                'revenue_ttm': latest.get('revenue_ttm', 0) / 1000,  # Billions
                'net_margin_ttm': latest.get('net_margin_ttm', 0),
                'pe_ttm': latest.get('pe_ttm', np.nan),
                'market_cap': latest.get('market_cap_billions', np.nan),
                'fcf_ttm': latest.get('freeCashFlow_ttm', 0) / 1000  # Billions
            })
    
    latest_df = pd.DataFrame(latest_data)
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MAG7 Latest Quarter Metrics Comparison', fontsize=16, fontweight='bold')
    
    # 1. Revenue TTM Bar Chart (Top-Left)
    if not latest_df.empty:
        bars1 = ax1.bar(latest_df['ticker'], latest_df['revenue_ttm'], 
                       color=[TICKER_COLORS[t] for t in latest_df['ticker']])
        ax1.set_title('Revenue TTM (Latest Quarter)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Revenue TTM ($B)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, latest_df['revenue_ttm']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'${value:.1f}B', ha='center', va='bottom')
    
    # 2. Net Margin TTM Bar Chart (Top-Right)
    if not latest_df.empty:
        bars2 = ax2.bar(latest_df['ticker'], latest_df['net_margin_ttm'], 
                       color=[TICKER_COLORS[t] for t in latest_df['ticker']])
        ax2.set_title('Net Margin TTM (Latest Quarter)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Net Margin TTM (%)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, latest_df['net_margin_ttm']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.1f}%', ha='center', va='bottom')
    
    # 3. P/E Ratio Bar Chart (Bottom-Left)
    if not latest_df.empty and not latest_df['pe_ttm'].isna().all():
        valid_pe = latest_df.dropna(subset=['pe_ttm'])
        # Filter out extreme P/E values
        valid_pe = valid_pe[(valid_pe['pe_ttm'] > 0) & (valid_pe['pe_ttm'] < 200)]
        if not valid_pe.empty:
            bars3 = ax3.bar(valid_pe['ticker'], valid_pe['pe_ttm'], 
                           color=[TICKER_COLORS[t] for t in valid_pe['ticker']])
            ax3.set_title('P/E TTM (Latest Quarter)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('P/E TTM')
            ax3.grid(True, alpha=0.3)
            
            # Set reasonable y-axis limits
            max_pe = valid_pe['pe_ttm'].max()
            ax3.set_ylim(0, min(100, max_pe * 1.2))
            
            # Add value labels on bars
            for bar, value in zip(bars3, valid_pe['pe_ttm']):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{value:.1f}', ha='center', va='bottom')
    
    # 4. Free Cash Flow TTM Bar Chart (Bottom-Right)
    if not latest_df.empty:
        bars4 = ax4.bar(latest_df['ticker'], latest_df['fcf_ttm'], 
                       color=[TICKER_COLORS[t] for t in latest_df['ticker']])
        ax4.set_title('Free Cash Flow TTM (Latest Quarter)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('FCF TTM ($B)')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars4, latest_df['fcf_ttm']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'${value:.1f}B', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'peer_valuation_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Created peer valuation metrics")

def main():
    """Main function for quarterly fundamentals processing."""
    print("📊 Step 6: Quarterly Fundamentals Processing")
    print("=" * 50)
    
    # Setup directories
    setup_directories()
    
    # Process each ticker
    all_data = []
    for ticker in TICKERS:
        print(f"\n🔄 Processing {ticker}...")
        
        # Load ticker data
        ticker_data = load_ticker_fundamentals(ticker)
        if not ticker_data:
            print(f"  ⚠️ No data loaded for {ticker}")
            continue
        
        # Process ticker data
        processed_df = process_ticker_data(ticker, ticker_data)
        if processed_df is not None and not processed_df.empty:
            all_data.append(processed_df)
            print(f"  ✅ Processed {ticker}: {len(processed_df)} records")
        else:
            print(f"  ⚠️ No processed data for {ticker}")
    
    if not all_data:
        print("❌ No data processed successfully")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(['source_ticker', 'periodEndDate']).reset_index(drop=True)
    
    print(f"\n✅ Combined dataset: {len(combined_df)} records")
    print(f"📅 Date range: {combined_df['periodEndDate'].min().strftime('%Y-%m-%d')} to {combined_df['periodEndDate'].max().strftime('%Y-%m-%d')}")
    
    # Save the dataset
    output_file = OUTPUT_DIR / 'fundamentals_quarterly_long.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"💾 Saved dataset: {output_file}")
    
    print(f"\n✅ Step 6 Complete!")

if __name__ == "__main__":
    main()
