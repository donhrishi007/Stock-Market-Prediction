"""
Step 4: Apple Peers Fundamental Data Collection using FMP API
============================================================

This module downloads fundamental data for Apple's key peers using the Financial Modeling Prep (FMP) API
and saves each dataset to separate Excel files for comprehensive analysis.

Target Companies:
- Microsoft (MSFT)
- Alphabet (GOOGL) 
- Amazon (AMZN)

Key Features:
1. Download annual and quarterly Income Statement, Balance Sheet, Cash Flow, and Key Metrics
2. Save each dataset into separate Excel files in structured subfolders
3. Clean and format data for analysis
4. Comprehensive error handling and progress tracking
5. Professional-grade data organization for stock prediction modeling

Data Structure:
- fmp_data/fundamentals/MSFT/annual/ & quarterly/
- fmp_data/fundamentals/GOOGL/annual/ & quarterly/
- fmp_data/fundamentals/AMZN/annual/ & quarterly/

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

# Target companies for fundamental analysis (Apple's key peers)
TARGET_COMPANIES = {
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc. (Google)',
    'AMZN': 'Amazon.com Inc.'
}


# Output base directory
OUTPUT_BASE_DIR = Path('fmp_data/fundamentals')

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
        response = requests.get(f"{FMP_BASE_URL}/profile/MSFT", params={'apikey': api_key})
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
        # Create directories for individual companies
        for ticker in TARGET_COMPANIES.keys():
            # Create annual and quarterly directories for each company
            annual_dir = OUTPUT_BASE_DIR / ticker / 'annual'
            quarterly_dir = OUTPUT_BASE_DIR / ticker / 'quarterly'
            
            annual_dir.mkdir(parents=True, exist_ok=True)
            quarterly_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"✅ Created directories for {ticker}:")
            print(f"   Annual: {annual_dir}")
            print(f"   Quarterly: {quarterly_dir}")
        
        
        return True
    except Exception as e:
        print(f"❌ Error creating output directories: {e}")
        return False

# ============================================================================
# FMP API FUNCTIONS
# ============================================================================

class FMPDataCollector:
    """Financial Modeling Prep API data collector for multiple companies."""
    
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

class PeersFundamentalProcessor:
    """Process fundamental data from FMP for Apple's peers and save to Excel files."""
    
    def __init__(self, fmp_collector):
        self.fmp = fmp_collector
        self.output_base_dir = OUTPUT_BASE_DIR
    
    def download_and_save_all_companies(self):
        """Download and save fundamental data for all target companies."""
        print("🏢 Downloading fundamental data for Apple's peers...")
        print(f"📊 Target companies: {', '.join(TARGET_COMPANIES.keys())}")
        
        results = {}
        
        for ticker, company_name in TARGET_COMPANIES.items():
            print(f"\n{'='*60}")
            print(f"🏢 Processing {ticker} - {company_name}")
            print(f"{'='*60}")
            
            try:
                # Download annual datasets
                print(f"\n📅 Downloading ANNUAL data for {ticker}...")
                annual_datasets = {
                    'income_statement': self.fmp.get_income_statement(ticker, period='annual'),
                    'balance_sheet': self.fmp.get_balance_sheet(ticker, period='annual'),
                    'cash_flow': self.fmp.get_cash_flow(ticker, period='annual'),
                    'key_metrics': self.fmp.get_key_metrics(ticker, period='annual')
                }
                
                # Process and save annual datasets
                annual_results = {}
                for dataset_name, data in annual_datasets.items():
                    if data:
                        result = self._process_and_save_dataset(ticker, dataset_name, data, 'annual')
                        annual_results[dataset_name] = result
                    else:
                        print(f"❌ Failed to download annual {dataset_name} for {ticker}")
                        annual_results[dataset_name] = False
                
                # Download quarterly datasets
                print(f"\n📅 Downloading QUARTERLY data for {ticker}...")
                quarterly_datasets = {
                    'income_statement': self.fmp.get_income_statement(ticker, period='quarterly'),
                    'balance_sheet': self.fmp.get_balance_sheet(ticker, period='quarterly'),
                    'cash_flow': self.fmp.get_cash_flow(ticker, period='quarterly'),
                    'key_metrics': self.fmp.get_key_metrics(ticker, period='quarterly')
                }
                
                # Process and save quarterly datasets
                quarterly_results = {}
                for dataset_name, data in quarterly_datasets.items():
                    if data:
                        result = self._process_and_save_dataset(ticker, dataset_name, data, 'quarterly')
                        quarterly_results[dataset_name] = result
                    else:
                        print(f"❌ Failed to download quarterly {dataset_name} for {ticker}")
                        quarterly_results[dataset_name] = False
                
                results[ticker] = {
                    'company_name': company_name,
                    'annual': annual_results,
                    'quarterly': quarterly_results
                }
                
                print(f"✅ Completed processing {ticker}")
                
            except Exception as e:
                print(f"❌ Error processing {ticker}: {e}")
                results[ticker] = {'error': str(e)}
        
        return results
    
    def _process_and_save_dataset(self, ticker, dataset_name, data, period_type):
        """Process and save a single dataset to Excel file."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if df.empty:
                print(f"⚠️ No data available for {period_type} {dataset_name} for {ticker}")
                return False
            
            # Clean and format the data
            df = self._clean_dataframe(df)
            
            # Choose output directory based on period type
            output_dir = self.output_base_dir / ticker / period_type
            
            # Create Excel file
            filename = f"{ticker.lower()}_{dataset_name}.xlsx"
            filepath = output_dir / filename
            
            # Save to Excel with formatting
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                sheet_name = dataset_name.replace('_', ' ').title()
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Get the workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets[sheet_name]
                
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
            
            print(f"✅ Saved {period_type} {dataset_name} for {ticker}: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing {period_type} {dataset_name} for {ticker}: {e}")
            return False
    
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
    
    def create_summary_report(self, results):
        """Create a summary report of downloaded data."""
        print("📋 Creating comprehensive summary report...")
        
        summary = {
            'download_date': datetime.now().isoformat(),
            'companies': list(TARGET_COMPANIES.keys()),
            'data_source': 'FMP API',
            'total_companies': len(TARGET_COMPANIES),
            'company_details': {}
        }
        
        # Process each company's results
        for ticker, company_data in results.items():
            if 'error' in company_data:
                summary['company_details'][ticker] = {
                    'status': 'error',
                    'error': company_data['error']
                }
                continue
            
            company_summary = {
                'company_name': company_data['company_name'],
                'status': 'success',
                'annual_files': 0,
                'quarterly_files': 0,
                'total_files': 0,
                'annual_data': [],
                'quarterly_data': []
            }
            
            # Count annual files
            for dataset_name, success in company_data['annual'].items():
                if success:
                    company_summary['annual_files'] += 1
                    company_summary['annual_data'].append(dataset_name)
            
            # Count quarterly files
            for dataset_name, success in company_data['quarterly'].items():
                if success:
                    company_summary['quarterly_files'] += 1
                    company_summary['quarterly_data'].append(dataset_name)
            
            company_summary['total_files'] = company_summary['annual_files'] + company_summary['quarterly_files']
            summary['company_details'][ticker] = company_summary
        
        # Save summary report
        summary_file = self.output_base_dir / "download_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Apple Peers Fundamental Data Download Summary\n")
            f.write(f"============================================\n\n")
            f.write(f"Download Date: {summary['download_date']}\n")
            f.write(f"Data Source: {summary['data_source']}\n")
            f.write(f"Total Companies: {summary['total_companies']}\n")
            f.write(f"Companies: {', '.join(summary['companies'])}\n\n")
            
            for ticker, details in summary['company_details'].items():
                f.write(f"{ticker} - {details.get('company_name', 'Unknown')}:\n")
                f.write(f"  Status: {details['status']}\n")
                if details['status'] == 'success':
                    f.write(f"  Total Files: {details['total_files']}\n")
                    f.write(f"  Annual Files: {details['annual_files']}\n")
                    f.write(f"  Quarterly Files: {details['quarterly_files']}\n")
                    f.write(f"  Annual Data: {', '.join(details['annual_data'])}\n")
                    f.write(f"  Quarterly Data: {', '.join(details['quarterly_data'])}\n")
                else:
                    f.write(f"  Error: {details['error']}\n")
                f.write(f"\n")
        
        print(f"📄 Summary report saved: {summary_file}")
        return summary
    

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function for Apple peers fundamental data collection."""
    print("🏢 Step 4: Apple Peers Fundamental Data Collection using FMP API")
    print("=" * 70)
    print("📊 Target Companies: Microsoft (MSFT), Alphabet (GOOGL), Amazon (AMZN)")
    print("📈 Data Types: Income Statement, Balance Sheet, Cash Flow, Key Metrics")
    print("📅 Periods: Annual and Quarterly")
    print("=" * 70)
    
    # Setup output directories
    if not setup_output_directories():
        print("❌ Failed to create output directories. Exiting.")
        return
    
    # Get API key
    print("\n🔑 Getting API key...")
    api_key = get_api_key()
    if not api_key:
        print("❌ No API key available. Exiting.")
        return
    
    print(f"✅ API key obtained: {api_key[:10]}...")
    
    # Test API connection
    print("\n🔍 Testing API connection...")
    if not test_api_connection(api_key):
        print("❌ API connection failed. Please check your API key.")
        return
    
    try:
        # Initialize FMP collector and processor
        fmp = FMPDataCollector(api_key)
        processor = PeersFundamentalProcessor(fmp)
        
        # Download and save all fundamental data
        print("\n📥 Downloading and saving fundamental data for all companies...")
        results = processor.download_and_save_all_companies()
        
        # Create summary report
        print("\n📋 Creating summary report...")
        summary = processor.create_summary_report(results)
        
        print("\n✅ Step 4 Complete!")
        print("\n📋 Summary:")
        print(f"  • Downloaded fundamental data for {len(TARGET_COMPANIES)} companies")
        print(f"  • Data saved in: {OUTPUT_BASE_DIR}")
        print(f"  • Companies processed: {', '.join(TARGET_COMPANIES.keys())}")
        
        # Print detailed results for companies
        for ticker, details in summary['company_details'].items():
            if details['status'] == 'success':
                print(f"\n📊 {ticker} ({details['company_name']}):")
                print(f"  • Total files: {details['total_files']}")
                print(f"  • Annual files: {details['annual_files']}")
                print(f"  • Quarterly files: {details['quarterly_files']}")
            else:
                print(f"\n❌ {ticker}: {details['error']}")
        
        print(f"\n📁 All files saved in: {OUTPUT_BASE_DIR.absolute()}")
        print(f"\n🎯 Ready for comprehensive peer analysis and stock prediction modeling!")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
