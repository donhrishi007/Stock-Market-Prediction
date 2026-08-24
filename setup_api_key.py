"""
API Key Setup Helper
====================

This script helps you set up your FMP API key for the stock prediction project.
It provides an interactive way to configure your API key securely.

Author: Finance ML Learning Project
Date: 2025
"""

import os
import sys
from pathlib import Path

def setup_api_key():
    """Interactive API key setup."""
    print("🔑 FMP API Key Setup Helper")
    print("=" * 40)
    print()
    print("This project uses the Financial Modeling Prep (FMP) API for data collection.")
    print("You need a free API key to run the data download scripts.")
    print()
    print("Get your free API key at: https://financialmodelingprep.com/developer/docs")
    print()
    
    # Check if API key is already set
    current_key = os.getenv('FMP_API_KEY')
    if current_key and current_key != 'YOUR_FMP_API_KEY_HERE':
        print(f"✅ API key is already set: {current_key[:10]}...")
        print("You can proceed with running the data download scripts.")
        return True
    
    print("No API key found. Let's set one up!")
    print()
    
    # Get API key from user
    api_key = input("Enter your FMP API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return False
    
    if len(api_key) < 20:
        print("⚠️  Warning: API key seems too short. Please verify it's correct.")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            return False
    
    # Show setup options
    print()
    print("Choose how to set up your API key:")
    print("1. Set environment variable (recommended)")
    print("2. Create fmp_config.py file")
    print("3. Both")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice in ['1', '3']:
        # Set environment variable
        print()
        print("To set the environment variable, run one of these commands:")
        print()
        print("Windows (Command Prompt):")
        print(f"set FMP_API_KEY={api_key}")
        print()
        print("Windows (PowerShell):")
        print(f"$env:FMP_API_KEY=\"{api_key}\"")
        print()
        print("Linux/Mac:")
        print(f"export FMP_API_KEY={api_key}")
        print()
        print("Or add it to your shell profile (.bashrc, .zshrc, etc.)")
    
    if choice in ['2', '3']:
        # Create config file
        config_content = f'''"""
FMP API Configuration
====================

This file contains the configuration for the Financial Modeling Prep (FMP) API.
Keep this file secure and never commit it to version control.

Author: Finance ML Learning Project
Date: 2025
"""

import os

# FMP API Configuration
FMP_API_KEY = "{api_key}"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
FMP_LIMIT = 120

# Validate API key
if FMP_API_KEY:
    print("✅ FMP API key is configured")
else:
    print("❌ FMP API key is not set")
'''
        
        config_path = Path('fmp_config.py')
        try:
            with open(config_path, 'w') as f:
                f.write(config_content)
            print(f"✅ Created {config_path}")
            print("⚠️  Remember to add fmp_config.py to your .gitignore file!")
        except Exception as e:
            print(f"❌ Error creating config file: {e}")
    
    print()
    print("🎉 Setup complete!")
    print("You can now run the data download scripts:")
    print("  python step1_data_downloader.py")
    print("  python step3_fmp_fundamental_data.py")
    print("  python step4_fundamentals.py")
    print("  python step5_macro_features.py")
    
    return True

if __name__ == "__main__":
    try:
        setup_api_key()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)
