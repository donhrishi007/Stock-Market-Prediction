"""
FMP API Configuration Template
=============================

This file contains the configuration for the Financial Modeling Prep (FMP) API.
To use this project, you need to:

1. Get a free API key from: https://financialmodelingprep.com/developer/docs
2. Replace 'YOUR_FMP_API_KEY_HERE' below with your actual API key
3. Keep this file secure and never commit it to version control

Alternative: Set the FMP_API_KEY environment variable instead of using this file.

Author: Finance ML Learning Project
Date: 2025
"""

import os

# FMP API Configuration
# Replace 'YOUR_FMP_API_KEY_HERE' with your actual FMP API key
FMP_API_KEY = os.getenv('FMP_API_KEY', 'YOUR_FMP_API_KEY_HERE')

# FMP API Base URL
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# FMP API Rate Limit (requests per minute)
FMP_LIMIT = 120

# Validate API key
if FMP_API_KEY == 'YOUR_FMP_API_KEY_HERE':
    print("⚠️  WARNING: Please set your FMP API key in fmp_config.py or as FMP_API_KEY environment variable")
    print("   Get your free API key at: https://financialmodelingprep.com/developer/docs")
