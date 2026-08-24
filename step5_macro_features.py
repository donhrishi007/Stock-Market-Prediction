"""
Step 5: US Macroeconomic Features Collection using FMP API
=========================================================

This module downloads key US macroeconomic indicators from the Financial Modeling Prep (FMP) API,
processes and aligns them by date, and creates visualizations for storytelling and analysis.

Macroeconomic Indicators:
- CPI (Consumer Price Index) - YoY % change
- GDP (US quarterly, annualized growth rate) - % growth
- Unemployment Rate - %
- Fed Funds Effective Rate - %
- 10-Year Treasury Yield - %

Key Features:
1. Download full historical data for all 5 indicators
2. Clean and align data by date (annual/quarterly)
3. Merge into master DataFrame with proper date indexing
4. Generate exploratory plots with event annotations
5. Save structured outputs for integration with stock data
6. Create events timeline for storytelling

Output Structure:
- artifacts/macro/macro_features.csv (tidy format)
- artifacts/macro/macro_features.xlsx (wide format)
- artifacts/macro/events.csv (economic events timeline)
- artifacts/macro/README.md (data documentation)
- stock_analysis/graphs/macro_step5/ (trend plots)

Author: Finance ML Learning Project
Date: 2025
"""

import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import FMP config, fallback to environment variable
try:
    from fmp_config import FMP_API_KEY, FMP_BASE_URL
except ImportError:
    FMP_API_KEY = os.getenv('FMP_API_KEY', 'YOUR_FMP_API_KEY_HERE')
    FMP_BASE_URL = "https://financialmodelingprep.com/stable"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Macroeconomic indicators to download
MACRO_INDICATORS = {
    'CPI': {
        'name': 'Consumer Price Index',
        'endpoint': 'economic-indicators',
        'param': 'CPI',
        'unit': '% YoY',
        'description': 'Consumer Price Index year-over-year percentage change'
    },
    'GDP': {
        'name': 'Gross Domestic Product',
        'endpoint': 'economic-indicators',
        'param': 'GDP',
        'unit': '% Growth',
        'description': 'US GDP quarterly annualized growth rate'
    },
    'UNEMPLOYMENT': {
        'name': 'Unemployment Rate',
        'endpoint': 'economic-indicators',
        'param': 'unemploymentRate',
        'unit': '%',
        'description': 'US unemployment rate'
    },
    'FED_FUNDS': {
        'name': 'Fed Funds Effective Rate',
        'endpoint': 'economic-indicators',
        'param': 'federalFunds',
        'unit': '%',
        'description': 'Federal funds effective rate'
    },
    'TREASURY_10Y': {
        'name': '10-Year Treasury Yield',
        'endpoint': 'treasury-rates',
        'param': None,  # Treasury rates endpoint doesn't need name parameter
        'unit': '%',
        'description': '10-year US Treasury bond yield'
    }
}

# Output directories
OUTPUT_BASE_DIR = Path('artifacts/macro')
GRAPHS_DIR = Path('stock_analysis/graphs/macro_step5')

# ============================================================================
# FMP API HELPER CLASS
# ============================================================================

class FMPMacroCollector:
    """Financial Modeling Prep API collector for macroeconomic data."""
    
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
    
    def make_request(self, endpoint, params=None):
        """Make API request with error handling."""
        if params is None:
            params = {}
        
        params['apikey'] = self.api_key
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed for {endpoint}: {str(e)}")
            return None
        except Exception as e:
            print(f"❌ Error processing {endpoint}: {str(e)}")
            return None
    
    def get_macro_indicator(self, indicator_name, endpoint, param_name):
        """Download macroeconomic indicator data."""
        print(f"📊 Downloading {indicator_name} data...")
        
        if endpoint == 'treasury-rates':
            # Treasury rates endpoint doesn't need name parameter
            data = self.make_request(endpoint, {})
        else:
            # Economic indicators endpoint needs name parameter
            data = self.make_request(endpoint, {'name': param_name})
        
        return data

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def setup_output_directories():
    """Create output directories for macro data and graphs."""
    try:
        # Create main output directory
        OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created macro data directory: {OUTPUT_BASE_DIR}")
        
        # Create graphs directory
        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created graphs directory: {GRAPHS_DIR}")
        
        return True
    except Exception as e:
        print(f"❌ Error creating output directories: {e}")
        return False

def process_macro_data(raw_data, indicator_key, indicator_info):
    """Process raw macro data into clean DataFrame."""
    if not raw_data:
        print(f"⚠️ No data available for {indicator_key}")
        return None
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(raw_data)
        
        if df.empty:
            print(f"⚠️ Empty DataFrame for {indicator_key}")
            return None
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        elif 'period' in df.columns:
            df['date'] = pd.to_datetime(df['period'], errors='coerce')
        else:
            print(f"⚠️ No date column found for {indicator_key}")
            return None
        
        # Find value column - handle different data structures
        value_col = None
        if indicator_key == 'TREASURY_10Y':
            # Treasury rates might have different column names
            for col in ['10Y', 'tenYear', 'rate', 'value', 'data']:
                if col in df.columns:
                    value_col = col
                    break
        else:
            # Economic indicators
            for col in ['value', 'data', 'rate', 'index']:
                if col in df.columns:
                    value_col = col
                    break
        
        if value_col is None:
            print(f"⚠️ No value column found for {indicator_key}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # Create clean DataFrame
        clean_df = pd.DataFrame({
            'date': df['date'],
            indicator_key.lower(): pd.to_numeric(df[value_col], errors='coerce')
        })
        
        # Remove rows with missing dates or values
        clean_df = clean_df.dropna()
        
        # Sort by date
        clean_df = clean_df.sort_values('date').reset_index(drop=True)
        
        # Forward fill missing values within series
        clean_df[indicator_key.lower()] = clean_df[indicator_key.lower()].fillna(method='ffill')
        
        # Filter out data before 1980 if too sparse
        clean_df = clean_df[clean_df['date'] >= '1980-01-01']
        
        print(f"✅ Processed {indicator_key}: {len(clean_df)} records from {clean_df['date'].min().strftime('%Y-%m-%d')} to {clean_df['date'].max().strftime('%Y-%m-%d')}")
        
        return clean_df
        
    except Exception as e:
        print(f"❌ Error processing {indicator_key}: {e}")
        return None

def merge_macro_data(macro_dataframes):
    """Merge all macro data into master DataFrame."""
    try:
        if not macro_dataframes:
            print("❌ No macro data to merge")
            return None
        
        # Start with first DataFrame
        master_df = macro_dataframes[0].copy()
        
        # Merge remaining DataFrames
        for df in macro_dataframes[1:]:
            if df is not None:
                master_df = pd.merge(master_df, df, on='date', how='outer')
        
        # Sort by date
        master_df = master_df.sort_values('date').reset_index(drop=True)
        
        # Add year column for alignment with fundamentals
        master_df['year'] = master_df['date'].dt.year
        
        # Forward fill missing values
        numeric_columns = master_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'year':
                master_df[col] = master_df[col].fillna(method='ffill')
        
        print(f"✅ Merged macro data: {len(master_df)} records, {len(numeric_columns)-1} indicators")
        print(f"📅 Date range: {master_df['date'].min().strftime('%Y-%m-%d')} to {master_df['date'].max().strftime('%Y-%m-%d')}")
        
        return master_df
        
    except Exception as e:
        print(f"❌ Error merging macro data: {e}")
        return None

def create_events_timeline():
    """Create economic events timeline for storytelling."""
    events_data = [
        {
            'event': 'Financial Crisis',
            'start_date': '2008-01-01',
            'end_date': '2009-12-31',
            'description': 'Global financial crisis and Great Recession'
        },
        {
            'event': 'COVID-19 Shock',
            'start_date': '2020-01-01',
            'end_date': '2020-06-30',
            'description': 'COVID-19 pandemic economic shock and lockdowns'
        },
        {
            'event': 'Fed Hiking Cycle',
            'start_date': '2022-01-01',
            'end_date': '2023-12-31',
            'description': 'Federal Reserve aggressive interest rate hiking cycle'
        },
        {
            'event': 'Dot-com Bubble',
            'start_date': '2000-01-01',
            'end_date': '2002-12-31',
            'description': 'Dot-com bubble burst and tech stock crash'
        },
        {
            'event': 'Great Recession Recovery',
            'start_date': '2010-01-01',
            'end_date': '2012-12-31',
            'description': 'Post-financial crisis recovery period'
        }
    ]
    
    events_df = pd.DataFrame(events_data)
    events_df['start_date'] = pd.to_datetime(events_df['start_date'])
    events_df['end_date'] = pd.to_datetime(events_df['end_date'])
    
    return events_df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_trend_plot(data, indicator_key, indicator_info, events_df=None):
    """Create trend plot for a macro indicator."""
    try:
        plt.figure(figsize=(12, 6))
        
        # Plot main data
        plt.plot(data['date'], data[indicator_key.lower()], 
                linewidth=2, color='#2E86AB', label=indicator_info['name'])
        
        # Add event annotations if provided
        if events_df is not None:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            for i, (_, event) in enumerate(events_df.iterrows()):
                plt.axvspan(event['start_date'], event['end_date'], 
                           alpha=0.2, color=colors[i % len(colors)], 
                           label=event['event'])
        
        # Formatting
        plt.title(f"{indicator_info['name']} ({data['date'].min().strftime('%Y')}–{data['date'].max().strftime('%Y')})", 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(f"{indicator_info['name']} ({indicator_info['unit']})", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        filename = f"{indicator_key.lower()}_trend.png"
        filepath = GRAPHS_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved trend plot: {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating plot for {indicator_key}: {e}")
        return False

def create_all_plots(master_df, events_df):
    """Create all macro trend plots."""
    print(f"\n📊 Creating trend plots...")
    
    plot_results = {}
    for indicator_key, indicator_info in MACRO_INDICATORS.items():
        if indicator_key.lower() in master_df.columns:
            result = create_trend_plot(master_df, indicator_key, indicator_info, events_df)
            plot_results[indicator_key] = result
        else:
            print(f"⚠️ No data available for {indicator_key} plot")
            plot_results[indicator_key] = False
    
    successful_plots = sum(1 for success in plot_results.values() if success)
    print(f"✅ Created {successful_plots}/{len(MACRO_INDICATORS)} trend plots")
    
    return plot_results

# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_macro_data(master_df, events_df):
    """Save macro data to CSV and Excel formats."""
    try:
        # Save CSV (tidy format)
        csv_file = OUTPUT_BASE_DIR / 'macro_features.csv'
        master_df.to_csv(csv_file, index=False)
        print(f"✅ Saved macro data (CSV): {csv_file}")
        
        # Save Excel (wide format with separate sheets)
        excel_file = OUTPUT_BASE_DIR / 'macro_features.xlsx'
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Main sheet with all data
            master_df.to_excel(writer, sheet_name='All_Indicators', index=False)
            
            # Individual sheets for each indicator
            for indicator_key in MACRO_INDICATORS.keys():
                if indicator_key.lower() in master_df.columns:
                    indicator_data = master_df[['date', 'year', indicator_key.lower()]].dropna()
                    sheet_name = indicator_key.replace('_', ' ').title()
                    indicator_data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ Saved macro data (Excel): {excel_file}")
        
        # Save events timeline
        events_file = OUTPUT_BASE_DIR / 'events.csv'
        events_df.to_csv(events_file, index=False)
        print(f"✅ Saved events timeline: {events_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving macro data: {e}")
        return False

def create_readme():
    """Create README documentation for macro data."""
    try:
        readme_content = """# US Macroeconomic Features Dataset

## Overview
This dataset contains key US macroeconomic indicators downloaded from the Financial Modeling Prep (FMP) API, processed and aligned for integration with stock fundamental and price data.

## Data Sources
- **API**: Financial Modeling Prep (FMP) API v4
- **Update Frequency**: As available from FMP
- **Date Range**: 1980-present (varies by indicator)

## Indicators

### 1. Consumer Price Index (CPI)
- **Variable**: `cpi`
- **Unit**: % YoY (Year-over-Year percentage change)
- **Frequency**: Monthly
- **Description**: Consumer Price Index year-over-year percentage change
- **Endpoint**: `/api/v4/economic?name=CPI`

### 2. Gross Domestic Product (GDP)
- **Variable**: `gdp`
- **Unit**: % Growth (Quarterly annualized growth rate)
- **Frequency**: Quarterly
- **Description**: US GDP quarterly annualized growth rate
- **Endpoint**: `/api/v4/economic?name=GDP`

### 3. Unemployment Rate
- **Variable**: `unemployment`
- **Unit**: % (Percentage)
- **Frequency**: Monthly
- **Description**: US unemployment rate
- **Endpoint**: `/api/v4/economic?name=Unemployment`

### 4. Fed Funds Effective Rate
- **Variable**: `fed_funds`
- **Unit**: % (Percentage)
- **Frequency**: Daily
- **Description**: Federal funds effective rate
- **Endpoint**: `/api/v4/economic?name=FEDFUNDS`

### 5. 10-Year Treasury Yield
- **Variable**: `treasury_10y`
- **Unit**: % (Percentage)
- **Frequency**: Daily
- **Description**: 10-year US Treasury bond yield
- **Endpoint**: `/api/v4/economic?name=10Y-Treasury-Yield`

## Data Processing

### Transformations Applied
1. **Date Alignment**: All indicators aligned to common date index
2. **Missing Value Handling**: Forward-fill within each series
3. **Data Filtering**: Removed sparse data before 1980
4. **Year Column**: Added for alignment with fiscal year data

### Data Quality
- Missing values are forward-filled within each series
- Data before 1980 filtered out if sparse
- All numeric values converted to appropriate types
- Date columns standardized to datetime format

## File Formats

### CSV Format (`macro_features.csv`)
- **Format**: Tidy/long format
- **Structure**: One row per date with all indicators
- **Use Case**: Easy merging with other datasets

### Excel Format (`macro_features.xlsx`)
- **Format**: Wide format with multiple sheets
- **Sheets**: 
  - `All_Indicators`: Complete dataset
  - Individual sheets for each indicator
- **Use Case**: Human-readable analysis and exploration

### Events Timeline (`events.csv`)
- **Format**: Event annotations for storytelling
- **Columns**: event, start_date, end_date, description
- **Use Case**: Adding context to visualizations and analysis

## Usage Examples

### Python/Pandas
```python
import pandas as pd

# Load macro data
macro_df = pd.read_csv('artifacts/macro/macro_features.csv')
macro_df['date'] = pd.to_datetime(macro_df['date'])

# Merge with stock data
stock_data = pd.read_csv('financial_data.csv')
stock_data['date'] = pd.to_datetime(stock_data['date'])

merged_data = pd.merge(stock_data, macro_df, on='date', how='left')
```

### Event Annotations
```python
# Load events for visualization
events_df = pd.read_csv('artifacts/macro/events.csv')
events_df['start_date'] = pd.to_datetime(events_df['start_date'])
events_df['end_date'] = pd.to_datetime(events_df['end_date'])

# Use in matplotlib plots
for _, event in events_df.iterrows():
    plt.axvspan(event['start_date'], event['end_date'], 
                alpha=0.2, label=event['event'])
```

## Integration Notes

### With Stock Price Data
- Use `date` column for exact date matching
- Consider forward-filling for daily stock data with monthly macro data

### With Fundamental Data
- Use `year` column for annual fundamental alignment
- Quarterly macro data can be aggregated to annual for fundamental analysis

### With Analysis Scripts
- All indicators are in percentage format (except raw values where noted)
- Missing values handled consistently across all series
- Ready for correlation analysis, regression modeling, and feature engineering

## Updates
- **Last Updated**: {update_date}
- **Script**: step5_macro_features.py
- **API Version**: FMP v4

## Notes
- Data quality varies by indicator and time period
- Some indicators may have gaps during market closures or data collection issues
- All transformations are documented and reproducible
""".format(update_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        readme_file = OUTPUT_BASE_DIR / 'README.md'
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✅ Created README: {readme_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating README: {e}")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function for macroeconomic features collection."""
    print("🏛️ Step 5: US Macroeconomic Features Collection using FMP API")
    print("=" * 80)
    print("📊 Indicators: CPI, GDP, Unemployment, Fed Funds, 10Y Treasury")
    print("📈 Outputs: CSV, Excel, Plots, Events Timeline, Documentation")
    print("=" * 80)
    
    # Setup output directories
    if not setup_output_directories():
        print("❌ Failed to create output directories. Exiting.")
        return
    
    try:
        # Initialize FMP collector
        fmp = FMPMacroCollector(FMP_API_KEY, FMP_BASE_URL)
        
        # Download and process all macro indicators
        print(f"\n📥 Downloading macroeconomic indicators...")
        macro_dataframes = []
        
        for indicator_key, indicator_info in MACRO_INDICATORS.items():
            # Download raw data
            raw_data = fmp.get_macro_indicator(indicator_info['name'], indicator_info['endpoint'], indicator_info['param'])
            
            # Process data
            processed_df = process_macro_data(raw_data, indicator_key, indicator_info)
            
            if processed_df is not None:
                macro_dataframes.append(processed_df)
        
        if not macro_dataframes:
            print("❌ No macro data downloaded. Exiting.")
            return
        
        # Merge all macro data
        print(f"\n🔄 Merging macro data...")
        master_df = merge_macro_data(macro_dataframes)
        
        if master_df is None:
            print("❌ Failed to merge macro data. Exiting.")
            return
        
        # Create events timeline
        print(f"\n📅 Creating events timeline...")
        events_df = create_events_timeline()
        
        # Create visualizations
        plot_results = create_all_plots(master_df, events_df)
        
        # Save all outputs
        print(f"\n💾 Saving outputs...")
        save_success = save_macro_data(master_df, events_df)
        readme_success = create_readme()
        
        # Final summary
        print(f"\n✅ Step 5 Complete!")
        print(f"\n📋 Summary:")
        print(f"  • Downloaded {len(macro_dataframes)} macroeconomic indicators")
        print(f"  • Date range: {master_df['date'].min().strftime('%Y-%m-%d')} to {master_df['date'].max().strftime('%Y-%m-%d')}")
        print(f"  • Indicators: {', '.join([k for k in MACRO_INDICATORS.keys() if k.lower() in master_df.columns])}")
        print(f"  • Plots created: {sum(1 for success in plot_results.values() if success)}/{len(MACRO_INDICATORS)}")
        print(f"  • Events timeline: {len(events_df)} major economic events")
        
        print(f"\n📁 Outputs saved in:")
        print(f"  • Data: {OUTPUT_BASE_DIR.absolute()}")
        print(f"  • Plots: {GRAPHS_DIR.absolute()}")
        
        print(f"\n🎯 Ready for integration with stock price and fundamental data!")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
