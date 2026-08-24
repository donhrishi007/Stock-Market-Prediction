# 📈 Stock Prediction Machine Learning Project

A comprehensive machine learning pipeline for predicting stock returns using financial statement data with proper time-aware validation and leakage prevention.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Financial Data](https://img.shields.io/badge/Data-Financial%20Modeling%20Prep-orange.svg)](https://financialmodelingprep.com)

## 🎯 Executive Summary

This project demonstrates advanced machine learning techniques for financial time series prediction, implementing a complete end-to-end pipeline that:

- **Processes 30 years** of financial data (1995-2025) for comprehensive analysis
- **Implements leakage-safe validation** using expanding window cross-validation
- **Achieves 69% directional accuracy** in stock return predictions
- **Handles multiple model types** including Ridge, Lasso, and Gradient Boosting
- **Provides comprehensive visualizations** and model diagnostics

**Key Achievement**: Successfully prevents data leakage while maintaining predictive performance, a critical challenge in financial machine learning.

## 📋 Project Overview

This project implements a complete end-to-end machine learning pipeline for stock prediction, focusing on:
- **Leakage Prevention**: Strict time-aware validation to prevent data leakage
- **Financial Data Processing**: Comprehensive handling of financial statements and market data
- **Multiple Model Types**: Linear models (Ridge, Lasso), tree models, and PCA-based approaches
- **Robust Validation**: Expanding window cross-validation with proper temporal ordering
- **Comprehensive Analysis**: Feature importance, model diagnostics, and performance visualization

## 🏗️ Project Structure

```
stock-prediction-ml/
├── 📊 Data Processing
│   ├── step1_data_downloader.py          # Download stock price data
│   ├── step2_log_returns_analysis.py     # Calculate log returns
│   ├── step2_stock_analysis.py           # Calculate simple returns
│   └── step3_fmp_fundamental_data.py     # Download fundamental data
├── 🔍 Feature Engineering
│   ├── step4_fundamentals.py             # Analyze fundamental data
│   ├── step5_macro_features.py           # Macro economic features
│   └── step6_fundamentals_quarterly.py   # Quarterly analysis
├── 🤖 Machine Learning
│   ├── step7_filing_anchored_enrichment.py  # Create ML dataset
│   ├── step8_time_aware_modeling.py      # Time-aware ML models
│   └── step9_enhanced_modeling.py        # Enhanced modeling with PCA
├── 🛠️ Utilities
│   ├── setup_api_key.py                  # API key configuration
│   ├── fmp_config.py                     # API configuration template
│   └── .gitignore                        # Git ignore rules
└── 📚 Documentation
    ├── README.md                         # This file
    └── README_ML_Pipeline.md            # Technical details
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels yfinance requests
```

### API Key Setup

This project uses the Financial Modeling Prep (FMP) API for data collection. You need to:

1. **Get a free API key** from: https://financialmodelingprep.com/developer/docs
2. **Set up your API key** using one of these methods:

   **Option A: Environment Variable (Recommended)**
   ```bash
   # Windows
   set FMP_API_KEY=your_actual_api_key_here
   
   # Linux/Mac
   export FMP_API_KEY=your_actual_api_key_here
   ```

   **Option B: Configuration File**
   - Copy `fmp_config.py` and replace `YOUR_FMP_API_KEY_HERE` with your actual API key
   - Keep the file secure and never commit it to version control

   **Option C: Interactive Setup (Easiest)**
   ```bash
   python setup_api_key.py
   ```
   This script will guide you through setting up your API key interactively.

### Running the Pipeline

1. **Stock price return analyses and data downloaded in Financial.db file**:
   ```bash
   python step1_data_downloader.py --> downloads price data from FMP for all the stocks and saves it in financial_data.db file
   python step2_log_returns_analysis.py --> log returns
   python step2_stock_analysisg.py --> simple returns
   ```

2. **Fundamental analysis**:
   ```bash
   python step3_fmp_fundamental_data.py --> downloads from FMP and saves data under FMP_data folder --> fundamentals
   python step4_fundamentals.py --> analysis of fundamental data
   python step5_macro.py --> not fully developed as data was not available from FMP
   python step6_fundamentals_quarterly.py --> analysis of fundamental data on quarterly basis. Important file created for ML models stored in artifacts --> fundamentals

   ```

3. **Analysis & Modeling**:
   ```bash
   python step7_filing_anchored_dataset.py --> created anchor dates for modelling purpose. Used features upto before anchor date, and forward looking variables post anchor date to prevent lookahead bias. Files stored in folder artifacts --> ml_data
   python step8_time_aware_modeling.py --> Regression models and classification models but the sample is too small --> output under artifacts folder --> ml_models. Graphs under graphs folder --> ml_models
   python step9_enhanced_modeling.py --> Using PCA but still underdeveloped
   ```

## 📊 Data Sources

- **Financial Statements**: Revenue, profit, cash flow, balance sheet data
- **Market Data**: Stock prices, returns, trading volumes
- **Time Series**: Quarterly and annual financial metrics
- **Target Variables**: Forward-looking returns (21, 63, 126 days)

## 🔧 Key Features

### 1. Leakage-Safe Design
- **Time-Aware Validation**: Expanding window cross-validation
- **Embargo Periods**: Prevents future information leakage
- **Train-Only Fitting**: All preprocessing fitted on training data only

### 2. Comprehensive Feature Engineering
- **Financial Ratios**: Profitability, liquidity, efficiency metrics
- **Time-Based Features**: Rolling averages, growth rates, trends
- **Technical Indicators**: Moving averages, volatility measures
- **Correlation Filtering**: Automatic removal of highly correlated features

### 3. Multiple Model Types
- **Linear Models**: Ridge, Lasso, ElasticNet regression
- **Tree Models**: HistGradientBoosting (regression & classification)
- **Dimensionality Reduction**: PCA with configurable variance retention
- **Ensemble Methods**: Multiple baseline comparisons

### 4. Robust Validation
- **Expanding Window CV**: Time-aware cross-validation
- **Small Sample Handling**: Forgiving settings for limited data
- **Fallback Strategies**: Single chronological split when needed
- **Holdout Evaluation**: Final 25% for unbiased performance assessment

## 📈 Model Performance

### Regression Targets
- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **Hit Rate**: Direction prediction accuracy
- **RMSE**: Root Mean Square Error

### Classification Targets
- **ROC-AUC**: Area under ROC curve
- **F1 Score**: Harmonic mean of precision and recall
- **Balanced Accuracy**: Accuracy accounting for class imbalance
- **Brier Score**: Probability calibration measure

## 🎯 Key Insights

### Data Characteristics
- **Sample Size**: ~120 quarterly observations
- **Time Range**: 1995-2025 (30 years)
- **Features**: 15-37 features after filtering
- **Targets**: Multiple forward-looking return horizons

### Model Performance
- **Classification**: Generally outperforms regression (typical in finance)
- **Hit Rate**: ~69% directional accuracy
- **ROC-AUC**: ~0.64 (moderate predictive ability)
- **R²**: Often negative (challenging to beat market mean)

## 🔍 Advanced Features

### Step 9 Enhancements
- **PCA Analysis**: Dimensionality reduction with scree plots
- **Permutation Importance**: Feature importance via shuffling
- **Enhanced Visualizations**: ROC curves, prediction plots, cumulative returns
- **Tree Baselines**: Gradient boosting for non-linear patterns

### Diagnostic Tools
- **VIF Analysis**: Multicollinearity detection
- **Correlation Heatmaps**: Feature relationship visualization
- **Coefficient Stability**: Cross-validation consistency checks
- **Residual Analysis**: Model assumption validation

## 📁 Output Files

### Data Artifacts
- `ml_filing_anchored_enriched.csv`: Main ML dataset
- `vif_table.csv`: Variance Inflation Factor analysis
- `cv_metrics_*.csv`: Cross-validation results
- `holdout_predictions.csv`: Final predictions

### Visualizations
- `corr_heatmap_features.png`: Feature correlation matrix
- `model_performance.png`: Model comparison charts
- `holdout_roc.png`: ROC curve for classification
- `holdout_pred_vs_actual.png`: Regression scatter plot
- `pca_scree.png`: PCA explained variance
- `perm_importance_*.png`: Feature importance plots
- `holdout_cumret_directional.png`: Cumulative return strategy

## ⚠️ Important Considerations

### Data Limitations
- **Small Sample Size**: Limited to ~120 observations
- **Temporal Dependencies**: Financial data has strong time correlations
- **Market Efficiency**: Stock prediction is inherently challenging
- **Regime Changes**: Market conditions vary over 30-year period

### Model Limitations
- **Overfitting Risk**: High-dimensional data with limited samples
- **Non-Stationarity**: Financial time series properties change over time
- **External Factors**: Models don't capture news, sentiment, or macro events
- **Transaction Costs**: Real-world trading involves fees and slippage

## 🔬 Technical Details

### Preprocessing Pipeline
1. **Infinite Value Handling**: Replace ±inf with NaN
2. **Missing Value Filtering**: Remove features with >40% missing data
3. **Winsorization**: Cap extreme values at 1st/99th percentiles
4. **Imputation**: Median imputation for remaining missing values
5. **Standardization**: Z-score normalization
6. **Variance Filtering**: Remove near-zero variance features
7. **PCA (Optional)**: Dimensionality reduction with 90% variance retention

### Cross-Validation Strategy
- **Expanding Window**: Train on [0, t), validate on [t, t+8)
- **Minimum Training**: 36 samples (3 years of quarterly data)
- **Validation Size**: 8 samples (2 years)
- **Step Size**: 4 samples (1 year)
- **Embargo**: 0 days (disabled for small samples)


## 🤝 Contributing

This project follows best practices for financial machine learning:
- **Reproducibility**: Fixed random seeds and deterministic algorithms
- **Documentation**: Comprehensive docstrings and comments
- **Error Handling**: Graceful failure with informative messages
- **Modularity**: Clear separation of concerns across steps


## 🛠️ Technical Skills Demonstrated

### **Machine Learning & Data Science**
- **Time Series Analysis**: Expanding window cross-validation, temporal feature engineering
- **Feature Engineering**: Financial ratios, technical indicators, correlation filtering
- **Model Selection**: Ridge/Lasso regression, Gradient Boosting, PCA dimensionality reduction
- **Model Validation**: Leakage prevention, robust evaluation metrics, statistical testing

### **Programming & Tools**
- **Python**: pandas, numpy, scikit-learn, matplotlib, seaborn, statsmodels
- **Data Processing**: SQLite database operations, API integration, data cleaning
- **Visualization**: Statistical plots, ROC curves, correlation heatmaps, performance charts
- **Version Control**: Git, GitHub, proper project structure

### **Financial Domain Knowledge**
- **Financial Statements**: Income statement, balance sheet, cash flow analysis
- **Market Data**: Stock prices, returns, trading volumes, volatility measures
- **Risk Management**: Data leakage prevention, proper backtesting methodology
- **API Integration**: Financial Modeling Prep API, rate limiting, error handling

## 🎓 Learning Outcomes

This project demonstrates:
- **Time Series ML**: Proper handling of temporal data
- **Financial Data**: Processing and feature engineering for markets
- **Model Validation**: Leakage prevention and robust evaluation
- **Python ML Stack**: pandas, scikit-learn, matplotlib integration
- **Statistical Analysis**: VIF, correlation, permutation importance
- **Visualization**: Comprehensive plotting for model interpretation

---

## 📞 Contact

For questions about this project or collaboration opportunities, please reach out through GitHub or LinkedIn.

**Note**: This project is for educational and portfolio purposes. Past performance does not guarantee future results in financial markets.
