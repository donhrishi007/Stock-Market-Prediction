# Stock Prediction ML Pipeline - Filing-Anchored Dataset

## Overview

This document describes the machine learning pipeline for stock prediction using a **filing-anchored approach** that eliminates look-ahead bias and creates a robust dataset for predictive modeling.

## 🎯 Key Innovation: Filing-Anchored Approach

Unlike traditional quarter-end aligned datasets, this pipeline uses **actual filing dates** as anchor points, ensuring:
- **No look-ahead bias**: Features use only data available before the filing date
- **Realistic trading scenarios**: Reflects when information actually becomes available to investors
- **Temporal accuracy**: Aligns with real-world decision-making timelines

## 📊 Dataset Structure

### Input Data Sources
- **Fundamental Data**: Quarterly financial statements from FMP API
- **Price Data**: Daily OHLCV data for AAPL and S&P 500 (^GSPC)
- **Filing Dates**: Actual SEC filing dates for each quarter

### Output Datasets

#### 1. `ml_filing_anchored_enriched.csv` (Full Dataset)
- **Records**: 120 quarterly observations (1995-2025)
- **Features**: 47 total columns
- **Purpose**: Complete dataset with all features and targets

#### 2. `ml_filing_anchored_focused.csv` (ML-Ready Dataset)
- **Records**: 120 quarterly observations
- **Features**: 25 key columns for machine learning
- **Purpose**: Streamlined dataset for model training

## 🔧 Feature Engineering

### Fundamental Features (Pre-Anchor)
These features are computed from financial statements and are available at filing time:

#### Revenue & Growth Metrics
- `revenue_yoy`: Year-over-year revenue growth
- `revenue_qoq`: Quarter-over-quarter revenue growth
- `revenue_ttm`: Trailing twelve months revenue

#### Profitability Metrics
- `gross_margin_ttm`: Gross profit margin (TTM)
- `op_margin_ttm`: Operating profit margin (TTM)
- `net_margin_ttm`: Net profit margin (TTM)
- `eps_yoy`: Year-over-year EPS growth

#### Cash Flow Metrics
- `freeCashFlow_yoy`: Year-over-year free cash flow growth
- `cash_to_debt`: Cash to debt ratio

#### Share Metrics
- `shares_chg_yoy`: Year-over-year shares outstanding change
- `pe_ttm`: Price-to-earnings ratio (TTM)

### Market Features (Pre-Anchor Only)
These features use **only historical price data before the anchor date**:

#### Momentum Features
- `mom_6m`: 6-month momentum (126 trading days)
- `mom_12m`: 12-month momentum (252 trading days)

#### Volatility Features
- `rv_21d`: 21-day realized volatility (annualized)
- `rv_63d`: 63-day realized volatility (annualized)

#### Risk Features
- `dd_6m`: 6-month maximum drawdown
- `dd_12m`: 12-month maximum drawdown

## 🎯 Target Variables (Forward-Looking)

All targets are computed from the anchor date forward:

### Log Returns
- `y_fwd_21d_log`: 21-day forward log return
- `y_fwd_63d_log`: 63-day forward log return  
- `y_fwd_126d_log`: 126-day forward log return

### Excess Returns (vs Market)
- `y_fwd_21d_excess`: 21-day excess return over S&P 500
- `y_fwd_63d_excess`: 63-day excess return over S&P 500
- `y_fwd_126d_excess`: 126-day excess return over S&P 500

### Binary Classification
- `y_up_63d`: Binary target for 63-day horizon (1 if positive return, 0 otherwise)

## 📈 Data Quality & Validation

### Temporal Integrity
- ✅ **No Look-Ahead Bias**: All features use only pre-anchor data
- ✅ **Filing Date Alignment**: Features computed as of actual filing dates
- ✅ **Trading Day Logic**: Targets computed using actual trading days

### Data Completeness
- **Coverage**: 30 years of data (1995-2025)
- **Frequency**: Quarterly observations
- **Missing Data**: Handled with appropriate NaN values

### Validation Checks
- Anchor date validation (must be after reported date)
- Feature computation verification
- Target calculation accuracy
- No data leakage confirmation

## 🚀 Usage for Machine Learning

### Primary Target
The **63-day forward log return** (`y_fwd_63d_log`) is recommended as the primary target for:
- Regression models
- Risk-adjusted return prediction
- Portfolio optimization

### Binary Classification
Use `y_up_63d` for:
- Directional prediction models
- Risk management systems
- Trading signal generation

### Feature Selection
Key features for model training:
1. **Fundamental**: `revenue_yoy`, `eps_yoy`, `gross_margin_ttm`
2. **Momentum**: `mom_6m`, `mom_12m`
3. **Volatility**: `rv_63d`
4. **Risk**: `dd_12m`

## 📁 File Structure

```
artifacts/
├── ml_data/
│   ├── ml_filing_anchored_enriched.csv    # Full dataset
│   └── ml_filing_anchored_focused.csv     # ML-ready dataset
├── fundamentals/
│   └── fundamentals_quarterly_long.csv    # Source fundamental data
graphs/
└── ml_targets/                            # Diagnostic plots
    ├── target_distributions.png
    ├── feature_distributions.png
    └── correlation_heatmap.png
```

## 🔄 Pipeline Steps

1. **Data Collection**: Download fundamental and price data
2. **Filing Anchoring**: Align data to actual filing dates
3. **Feature Engineering**: Compute fundamental and market features
4. **Target Creation**: Generate forward-looking targets
5. **Validation**: Ensure no look-ahead bias
6. **Output**: Create ML-ready datasets

## ⚠️ Important Notes

### Temporal Constraints
- Features use data **before** anchor date only
- Targets use data **after** anchor date only
- No overlapping time periods

### Trading Considerations
- All dates are trading days (excludes weekends/holidays)
- Returns are log returns for mathematical properties
- Volatility is annualized for comparability

### Model Development
- Use cross-validation with temporal splits
- Avoid future data leakage
- Consider market regime changes
- Validate on out-of-sample data

## 📊 Sample Data

```python
import pandas as pd

# Load the ML-ready dataset
df = pd.read_csv('artifacts/ml_data/ml_filing_anchored_focused.csv')

# Check basic statistics
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['anchor_date'].min()} to {df['anchor_date'].max()}")

# View key features
feature_cols = ['mom_6m', 'mom_12m', 'rv_63d', 'y_fwd_63d_log']
print(df[feature_cols].describe())
```

## 🎯 Next Steps

1. **Exploratory Data Analysis**: Analyze feature distributions and correlations
2. **Feature Engineering**: Create additional derived features if needed
3. **Model Selection**: Test various ML algorithms (Random Forest, XGBoost, Neural Networks)
4. **Validation**: Implement proper temporal cross-validation
5. **Backtesting**: Test strategies on historical data
6. **Deployment**: Implement live trading system

---

**Created**: 2025
**Author**: Finance ML Learning Project
**Status**: Ready for ML model development
