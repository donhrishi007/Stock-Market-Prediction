"""
Stock Data Downloader Module - STEP 1 (FMP API Version)
======================================================

🚀 EXECUTION ORDER: RUN THIS FILE FIRST!

This module handles downloading stock price data for:
- MAG7 (Magnificent 7) stocks: AAPL, MSFT, AMZN, GOOGL, NVDA, TSLA, META
- S&P 500 index (^GSPC)

The data is downloaded from Financial Modeling Prep (FMP) API and stored in a SQLite database.

Author: Finance ML Learning Project
Date: 2025
"""

import requests
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Import FMP configuration
try:
    from fmp_config import FMP_API_KEY, FMP_BASE_URL
except ImportError:
    # Fallback configuration - use environment variable
    FMP_API_KEY = os.getenv('FMP_API_KEY', 'YOUR_FMP_API_KEY_HERE')
    FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# Define the MAG7 stocks we want to download
MAG7_TICKERS = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META']

# S&P 500 ticker (FMP uses different format)
SP500_TICKER = '^GSPC'

# Database configuration
DB_NAME = 'financial_data.db'

# Date range for data download (adjust as needed)
START_DATE = '1990-01-01'  # Start from 1990 for comprehensive historical data
END_DATE = datetime.now().strftime('%Y-%m-%d')  # Current date

# ============================================================================
# FMP API FUNCTIONS
# ============================================================================

def test_fmp_connection():
    """Test FMP API connection."""
    print("🔍 Testing FMP API connection...")
    
    try:
        # Test with Apple stock
        url = f"{FMP_BASE_URL}/profile/AAPL"
        params = {'apikey': FMP_API_KEY}
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                print("✅ FMP API connection successful!")
                print(f"   Company: {data[0].get('companyName', 'Unknown')}")
                return True
            else:
                print("❌ API returned empty data")
                return False
        else:
            print(f"❌ API request failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API connection test failed: {e}")
        return False

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def create_database():
    """Create SQLite database for storing stock data."""
    print("🗄️ Creating database...")
    
    db_path = os.path.join(os.path.dirname(__file__), DB_NAME)
    conn = sqlite3.connect(db_path)
    conn.close()
    
    print(f"✅ Database created: {DB_NAME}")
    return db_path

def get_database_connection():
    """Get connection to the SQLite database."""
    db_path = os.path.join(os.path.dirname(__file__), DB_NAME)
    return sqlite3.connect(db_path)

# ============================================================================
# DATA DOWNLOAD FUNCTIONS
# ============================================================================

def download_stock_data(ticker, start_date=START_DATE, end_date=END_DATE):
    """
    Download stock data for a specific ticker using FMP API.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', '^GSPC')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        pd.DataFrame: Stock data with OHLCV + Adj_Close columns
    """
    print(f"📥 Downloading {ticker} data from FMP API ({start_date} to {end_date})...")
    
    try:
        # FMP API endpoint for historical price data
        url = f"{FMP_BASE_URL}/historical-price-full/{ticker}"
        
        # API parameters
        params = {
            'apikey': FMP_API_KEY,
            'from': start_date,
            'to': end_date
        }
        
        # Make API request
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Parse JSON response
        data_json = response.json()
        
        if 'historical' not in data_json:
            print(f"⚠️ No historical data found for {ticker}")
            return None
        
        # Convert to DataFrame
        historical_data = data_json['historical']
        if not historical_data:
            print(f"⚠️ No data found for {ticker}")
            return None
        
        # Create DataFrame
        data = pd.DataFrame(historical_data)
        
        # Rename and reorder columns to match yfinance format
        column_mapping = {
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adjClose': 'Adj_Close',  # Add adjusted close
            'volume': 'Volume'
        }
        
        # Select and rename columns
        data = data[list(column_mapping.keys())].rename(columns=column_mapping)
        
        # Convert Date to datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Sort by date (oldest first, then we'll reverse)
        data = data.sort_values('Date', ascending=True)
        
        # Convert numeric columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove any rows with NaN values
        data = data.dropna()
        
        print(f"✅ Downloaded {len(data)} records for {ticker}")
        print(f"📅 Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed for {ticker}: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ Error downloading {ticker}: {str(e)}")
        return None

def save_data_to_database(data, ticker):
    """
    Save stock data to SQLite database.
    
    Args:
        data (pd.DataFrame): Stock data to save
        ticker (str): Stock ticker symbol
    """
    if data is None or data.empty:
        print(f"⚠️ No data to save for {ticker}")
        return
    
    print(f"💾 Saving {ticker} data to database...")
    
    try:
        conn = get_database_connection()
        
        # Create table name
        table_name = f'{ticker}_price'
        
        # Save data to database
        data.to_sql(table_name, conn, if_exists='replace', index=False)
        
        conn.close()
        
        print(f"✅ {ticker} data saved to table: {table_name}")
        
    except Exception as e:
        print(f"❌ Error saving {ticker} data: {str(e)}")

def download_and_save_stock(ticker, start_date=START_DATE, end_date=END_DATE):
    """
    Download and save stock data for a single ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    """
    print(f"\n🔄 Processing {ticker}...")
    
    # Download data
    data = download_stock_data(ticker, start_date, end_date)
    
    if data is not None:
        # Save to database
        save_data_to_database(data, ticker)
        
        # Add a small delay to be respectful to the API
        time.sleep(1)
    else:
        print(f"❌ Failed to download {ticker}")

# ============================================================================
# BATCH DOWNLOAD FUNCTIONS
# ============================================================================

def download_mag7_data(start_date=START_DATE, end_date=END_DATE):
    """
    Download data for all MAG7 stocks.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    """
    print("🚀 Starting MAG7 stocks data download...")
    print("=" * 50)
    
    for ticker in MAG7_TICKERS:
        download_and_save_stock(ticker, start_date, end_date)
    
    print("\n✅ MAG7 stocks download complete!")

def download_sp500_data(start_date=START_DATE, end_date=END_DATE):
    """
    Download S&P 500 index data.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    """
    print("📊 Downloading S&P 500 data...")
    print("=" * 30)
    
    download_and_save_stock(SP500_TICKER, start_date, end_date)
    
    print("✅ S&P 500 download complete!")

def download_all_data(start_date=START_DATE, end_date=END_DATE):
    """
    Download all required data (MAG7 + S&P 500).
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    """
    print("🌍 Starting comprehensive data download...")
    print("=" * 60)
    
    # Create database if it doesn't exist
    create_database()
    
    # Download MAG7 stocks
    download_mag7_data(start_date, end_date)
    
    # Download S&P 500
    download_sp500_data(start_date, end_date)
    
    print("\n🎉 All data download complete!")
    print(f"📊 Database: {DB_NAME}")
    print(f"📅 Date range: {start_date} to {end_date}")
    print(f"📈 Stocks downloaded: {len(MAG7_TICKERS)} MAG7 + S&P 500")

# ============================================================================
# DATA VERIFICATION FUNCTIONS
# ============================================================================

def verify_downloaded_data():
    """Verify that all required data has been downloaded successfully."""
    print("🔍 Verifying downloaded data...")
    print("=" * 40)
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]
    
    print(f"📊 Found {len(table_names)} tables in database:")
    
    all_tickers = MAG7_TICKERS + [SP500_TICKER]
    missing_tickers = []
    
    for ticker in all_tickers:
        table_name = f'{ticker}_price'
        if table_name in table_names:
            # Get record count (handle special characters in table names)
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            count = cursor.fetchone()[0]
            
            # Get date range (handle special characters in table names)
            cursor.execute(f'SELECT MIN(Date), MAX(Date) FROM "{table_name}"')
            date_range = cursor.fetchone()
            
            print(f"  ✅ {ticker}: {count} records ({date_range[0]} to {date_range[1]})")
        else:
            print(f"  ❌ {ticker}: Missing")
            missing_tickers.append(ticker)
    
    conn.close()
    
    if missing_tickers:
        print(f"\n⚠️ Missing data for: {', '.join(missing_tickers)}")
        return False
    else:
        print(f"\n✅ All data verified successfully!")
        return True

def get_data_summary():
    """Get a summary of all downloaded data."""
    print("📋 Data Summary")
    print("=" * 30)
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    all_tickers = MAG7_TICKERS + [SP500_TICKER]
    
    for ticker in all_tickers:
        table_name = f'{ticker}_price'
        
        try:
            # Get basic stats (handle special characters in table names)
            cursor.execute(f'SELECT COUNT(*), MIN(Date), MAX(Date) FROM "{table_name}"')
            count, min_date, max_date = cursor.fetchone()
            
            # Get price range (handle special characters in table names)
            cursor.execute(f'SELECT MIN(Close), MAX(Close) FROM "{table_name}"')
            min_price, max_price = cursor.fetchone()
            
            # Get adjusted close range (handle special characters in table names)
            cursor.execute(f'SELECT MIN(Adj_Close), MAX(Adj_Close) FROM "{table_name}"')
            min_adj_price, max_adj_price = cursor.fetchone()
            
            print(f"\n📈 {ticker}:")
            print(f"  Records: {count:,}")
            print(f"  Date Range: {min_date} to {max_date}")
            print(f"  Close Price Range: ${min_price:.2f} - ${max_price:.2f}")
            print(f"  Adj Close Price Range: ${min_adj_price:.2f} - ${max_adj_price:.2f}")
            
        except Exception as e:
            print(f"  ❌ Error reading {ticker}: {str(e)}")
    
    conn.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function for data downloading."""
    print("🚀 Stock Data Downloader (FMP API Version)")
    print("=" * 60)
    
    # Test FMP API connection first
    if not test_fmp_connection():
        print("❌ FMP API connection failed. Please check your API key.")
        return
    
    # Download all data
    download_all_data()
    
    # Verify the download
    verify_downloaded_data()
    
    # Show summary
    get_data_summary()

if __name__ == "__main__":
    main()
