"""
Step 3: Apple Fundamental Data Collection using FMP API
======================================================

This module downloads Apple's fundamental data using the Financial Modeling Prep (FMP) API
and saves each dataset to separate Excel files for easy analysis.

Key Features:
1. Download annual Income Statement, Balance Sheet, Cash Flow, and Key Metrics for Apple (AAPL)
2. Save each dataset into a separate Excel file in "fmp_data/apple" subfolder
3. Clean and format data for analysis
4. Comprehensive error handling and progress tracking

Author: Finance ML Learning Project
Date: 2025
"""

import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Import configuration
try:
    from fmp_config import FMP_API_KEY, FMP_BASE_URL, FMP_LIMIT
except ImportError:
    # Fallback configuration - use environment variable
    FMP_API_KEY = os.getenv('FMP_API_KEY', 'YOUR_FMP_API_KEY_HERE')
    FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
    FMP_LIMIT = 120

# Target stock for fundamental analysis (Apple only)
TARGET_STOCK = 'AAPL'

# Output directories for Excel files
OUTPUT_DIR_ANNUAL = Path('fmp_data/apple/annual')
OUTPUT_DIR_QUARTERLY = Path('fmp_data/apple/quarterly')

# ============================================================================
# CONFIGURATION HELPER
# ============================================================================

def get_api_key():
    """Get FMP API key from various sources."""
    # Try environment variable first
    api_key = os.getenv('FMP_API_KEY')
    if api_key:
        print(f"🔑 Using API key from environment variable")
        return api_key
    
    # Use the provided API key
    print(f"🔍 Checking hardcoded API key: {FMP_API_KEY[:10] if FMP_API_KEY else 'None'}...")
    if FMP_API_KEY:
        print(f"🔑 Using hardcoded API key")
        return FMP_API_KEY
    
    print("❌ No API key found in environment or hardcoded value")
    return None

def test_api_connection(api_key):
    """Test FMP API connection."""
    print("🔍 Testing FMP API connection...")
    
    try:
        response = requests.get(f"{FMP_BASE_URL}/profile/AAPL", params={'apikey': api_key})
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
# DIRECTORY SETUP
# ============================================================================

def setup_output_directories():
    """Create output directories for Excel files."""
    try:
        OUTPUT_DIR_ANNUAL.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR_QUARTERLY.mkdir(parents=True, exist_ok=True)
        print(f"✅ Annual data directory created: {OUTPUT_DIR_ANNUAL}")
        print(f"✅ Quarterly data directory created: {OUTPUT_DIR_QUARTERLY}")
        return True
    except Exception as e:
        print(f"❌ Error creating output directories: {e}")
        return False

# ============================================================================
# FMP API FUNCTIONS
# ============================================================================

class FMPDataCollector:
    """Financial Modeling Prep API data collector."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = FMP_BASE_URL
        self.session = requests.Session()
        
    def make_request(self, endpoint, params=None):
        """Make API request to FMP."""
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return None
    
    def get_income_statement(self, symbol, limit=None, period='annual'):
        """Get income statement data."""
        if limit is None:
            limit = FMP_LIMIT
        period_type = "quarterly" if period == 'quarterly' else "annual"
        print(f"📊 Downloading {period_type} income statement for {symbol}...")
        data = self.make_request(f"income-statement/{symbol}", {"limit": limit, "period": period})
        return data
    
    def get_balance_sheet(self, symbol, limit=None, period='annual'):
        """Get balance sheet data."""
        if limit is None:
            limit = FMP_LIMIT
        period_type = "quarterly" if period == 'quarterly' else "annual"
        print(f"📊 Downloading {period_type} balance sheet for {symbol}...")
        data = self.make_request(f"balance-sheet-statement/{symbol}", {"limit": limit, "period": period})
        return data
    
    def get_cash_flow(self, symbol, limit=None, period='annual'):
        """Get cash flow statement data."""
        if limit is None:
            limit = FMP_LIMIT
        period_type = "quarterly" if period == 'quarterly' else "annual"
        print(f"📊 Downloading {period_type} cash flow statement for {symbol}...")
        data = self.make_request(f"cash-flow-statement/{symbol}", {"limit": limit, "period": period})
        return data
    
    def get_financial_ratios(self, symbol, limit=None, period='annual'):
        """Get financial ratios data."""
        if limit is None:
            limit = FMP_LIMIT
        period_type = "quarterly" if period == 'quarterly' else "annual"
        print(f"📊 Downloading {period_type} financial ratios for {symbol}...")
        data = self.make_request(f"ratios/{symbol}", {"limit": limit, "period": period})
        return data
    
    def get_key_metrics(self, symbol, limit=None, period='annual'):
        """Get key metrics data."""
        if limit is None:
            limit = FMP_LIMIT
        period_type = "quarterly" if period == 'quarterly' else "annual"
        print(f"📊 Downloading {period_type} key metrics for {symbol}...")
        data = self.make_request(f"key-metrics/{symbol}", {"limit": limit, "period": period})
        return data
    
    def get_company_profile(self, symbol):
        """Get company profile information."""
        print(f"📊 Downloading company profile for {symbol}...")
        data = self.make_request(f"profile/{symbol}")
        return data

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

class AppleFundamentalProcessor:
    """Process Apple fundamental data from FMP and save to Excel files."""
    
    def __init__(self, fmp_collector):
        self.fmp = fmp_collector
        self.output_dir_annual = OUTPUT_DIR_ANNUAL
        self.output_dir_quarterly = OUTPUT_DIR_QUARTERLY
    
    def download_and_save_fundamental_data(self):
        """Download and save Apple's fundamental data to Excel files."""
        print("🍎 Downloading Apple fundamental data from FMP...")
        
        # Download annual datasets
        print("\n📅 Downloading ANNUAL data...")
        annual_datasets = {
            'income_statement': self.fmp.get_income_statement(TARGET_STOCK, period='annual'),
            'balance_sheet': self.fmp.get_balance_sheet(TARGET_STOCK, period='annual'),
            'cash_flow': self.fmp.get_cash_flow(TARGET_STOCK, period='annual'),
            'key_metrics': self.fmp.get_key_metrics(TARGET_STOCK, period='annual')
        }
        
        # Process and save annual datasets
        for dataset_name, data in annual_datasets.items():
            if data:
                self._process_and_save_dataset(dataset_name, data, 'annual')
            else:
                print(f"❌ Failed to download annual {dataset_name}")
        
        # Download quarterly datasets
        print("\n📅 Downloading QUARTERLY data...")
        quarterly_datasets = {
            'income_statement': self.fmp.get_income_statement(TARGET_STOCK, period='quarterly'),
            'balance_sheet': self.fmp.get_balance_sheet(TARGET_STOCK, period='quarterly'),
            'cash_flow': self.fmp.get_cash_flow(TARGET_STOCK, period='quarterly'),
            'key_metrics': self.fmp.get_key_metrics(TARGET_STOCK, period='quarterly')
        }
        
        # Process and save quarterly datasets
        for dataset_name, data in quarterly_datasets.items():
            if data:
                self._process_and_save_dataset(dataset_name, data, 'quarterly')
            else:
                print(f"❌ Failed to download quarterly {dataset_name}")
        
        print("✅ All datasets processed and saved to Excel files")
    
    def _process_and_save_dataset(self, dataset_name, data, period_type):
        """Process and save a single dataset to Excel file."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if df.empty:
                print(f"⚠️ No data available for {period_type} {dataset_name}")
                return
            
            # Clean and format the data
            df = self._clean_dataframe(df)
            
            # Choose output directory based on period type
            output_dir = self.output_dir_annual if period_type == 'annual' else self.output_dir_quarterly
            
            # Create Excel file
            filename = f"apple_{dataset_name}.xlsx"
            filepath = output_dir / filename
            
            # Save to Excel with formatting
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=dataset_name.replace('_', ' ').title(), index=False)
                
                # Get the workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets[dataset_name.replace('_', ' ').title()]
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"✅ Saved {period_type} {dataset_name}: {filepath}")
            
        except Exception as e:
            print(f"❌ Error processing {period_type} {dataset_name}: {e}")
    
    def _clean_dataframe(self, df):
        """Clean and format DataFrame for better readability."""
        # Convert date columns to datetime
        date_columns = ['date', 'calendarYear', 'filingDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Sort by date if available
        if 'date' in df.columns:
            df = df.sort_values('date', ascending=False)
        elif 'calendarYear' in df.columns:
            df = df.sort_values('calendarYear', ascending=False)
        
        # Format numeric columns (convert to millions for large numbers)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['calendarYear', 'period']:  # Don't format year columns
                # Convert very large numbers to millions for readability
                if df[col].abs().max() > 1e9:
                    df[col] = df[col] / 1e6
                    df.rename(columns={col: f"{col} (Millions)"}, inplace=True)
        
        # Replace NaN values with empty strings for better Excel display
        df = df.fillna('')
        
        return df
    
    def create_summary_report(self):
        """Create a summary report of downloaded data."""
        print("📋 Creating summary report...")
        
        # List all Excel files in both directories
        annual_files = list(self.output_dir_annual.glob("*.xlsx"))
        quarterly_files = list(self.output_dir_quarterly.glob("*.xlsx"))
        all_files = annual_files + quarterly_files
        
        summary = {
            'download_date': datetime.now().isoformat(),
            'ticker': TARGET_STOCK,
            'data_source': 'FMP API',
            'annual_files': len(annual_files),
            'quarterly_files': len(quarterly_files),
            'total_files': len(all_files),
            'annual_data': [],
            'quarterly_data': []
        }
        
        # Process annual files
        for file in annual_files:
            try:
                file_info = self._get_file_info(file)
                summary['annual_data'].append(file_info)
            except Exception as e:
                print(f"⚠️ Error reading {file.name}: {e}")
        
        # Process quarterly files
        for file in quarterly_files:
            try:
                file_info = self._get_file_info(file)
                summary['quarterly_data'].append(file_info)
            except Exception as e:
                print(f"⚠️ Error reading {file.name}: {e}")
        
        # Save summary report
        summary_file = self.output_dir_annual.parent / "download_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Apple Fundamental Data Download Summary\n")
            f.write(f"=====================================\n\n")
            f.write(f"Download Date: {summary['download_date']}\n")
            f.write(f"Ticker: {summary['ticker']}\n")
            f.write(f"Data Source: {summary['data_source']}\n")
            f.write(f"Total Files Created: {summary['total_files']}\n")
            f.write(f"  - Annual Files: {summary['annual_files']}\n")
            f.write(f"  - Quarterly Files: {summary['quarterly_files']}\n\n")
            
            f.write("ANNUAL DATA:\n")
            f.write("============\n")
            for file_info in summary['annual_data']:
                f.write(f"• {file_info['filename']}\n")
                f.write(f"  - Rows: {file_info['rows']}\n")
                f.write(f"  - Columns: {file_info['columns']}\n")
                f.write(f"  - Size: {file_info['size_bytes']:,} bytes\n")
                f.write(f"  - Date Range: {file_info['date_range']}\n\n")
            
            f.write("QUARTERLY DATA:\n")
            f.write("===============\n")
            for file_info in summary['quarterly_data']:
                f.write(f"• {file_info['filename']}\n")
                f.write(f"  - Rows: {file_info['rows']}\n")
                f.write(f"  - Columns: {file_info['columns']}\n")
                f.write(f"  - Size: {file_info['size_bytes']:,} bytes\n")
                f.write(f"  - Date Range: {file_info['date_range']}\n\n")
        
        print(f"📄 Summary report saved: {summary_file}")
        return summary
    
    def _get_file_info(self, file):
        """Get file information for summary report."""
        file_size = file.stat().st_size
        df = pd.read_excel(file)
        return {
            'filename': file.name,
            'size_bytes': file_size,
            'rows': len(df),
            'columns': len(df.columns),
            'date_range': self._get_date_range(df)
        }
    
    def _get_date_range(self, df):
        """Get date range from DataFrame."""
        date_columns = ['date', 'calendarYear', 'filingDate']
        for col in date_columns:
            if col in df.columns:
                try:
                    dates = pd.to_datetime(df[col], errors='coerce')
                    valid_dates = dates.dropna()
                    if not valid_dates.empty:
                        return f"{valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}"
                except:
                    pass
        return "N/A"

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function for FMP fundamental data collection."""
    print("🍎 Step 3: Apple Fundamental Data Collection using FMP API")
    print("=" * 70)
    
    # Setup output directories
    if not setup_output_directories():
        print("❌ Failed to create output directories. Exiting.")
        return
    
    # Get API key
    print("🔑 Getting API key...")
    api_key = get_api_key()
    if not api_key:
        print("❌ No API key available. Exiting.")
        return
    
    print(f"✅ API key obtained: {api_key[:10]}...")
    
    # Test API connection
    print("🔍 Testing API connection...")
    if not test_api_connection(api_key):
        print("❌ API connection failed. Please check your API key.")
        return
    
    try:
        # Initialize FMP collector and processor
        fmp = FMPDataCollector(api_key)
        processor = AppleFundamentalProcessor(fmp)
        
        # Download and save all fundamental data
        print("\n📥 Downloading and saving Apple fundamental data...")
        processor.download_and_save_fundamental_data()
        
        # Create summary report
        print("\n📋 Creating summary report...")
        summary = processor.create_summary_report()
        
        print("\n✅ Step 3 Complete!")
        print("\n📋 Summary:")
        print(f"  • Downloaded Apple fundamental data using FMP API")
        print(f"  • Annual data saved in: {OUTPUT_DIR_ANNUAL}")
        print(f"  • Quarterly data saved in: {OUTPUT_DIR_QUARTERLY}")
        print(f"  • Total files created: {summary['total_files']}")
        print(f"    - Annual files: {summary['annual_files']}")
        print(f"    - Quarterly files: {summary['quarterly_files']}")
        
        # Print annual file details
        print(f"\n📅 Annual Data:")
        for file_info in summary['annual_data']:
            print(f"  • {file_info['filename']}: {file_info['rows']} rows, {file_info['columns']} columns")
        
        # Print quarterly file details
        print(f"\n📅 Quarterly Data:")
        for file_info in summary['quarterly_data']:
            print(f"  • {file_info['filename']}: {file_info['rows']} rows, {file_info['columns']} columns")
        
        print(f"\n📁 All files saved in: {OUTPUT_DIR_ANNUAL.parent.absolute()}")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
