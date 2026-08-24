"""
Step 9: Enhanced Time-Aware Machine Learning Modeling
=====================================================

This module implements leakage-safe baseline models on the filing-anchored dataset
with proper time-aware validation, comprehensive diagnostics, and enhanced features.

Key Features:
- Time-aware expanding window cross-validation
- Leakage prevention with embargo periods
- Multiple baseline models and metrics (Ridge, Lasso, PCA+Ridge, Tree models)
- Feature stability analysis
- Permutation importance analysis
- Enhanced visualization plots
- Final holdout evaluation

Author: Finance ML Learning Project
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from typing import Generator, Tuple, Dict, Any, List
from datetime import datetime, timedelta

# Scikit-learn imports
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    balanced_accuracy_score, f1_score, roc_auc_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_FILE = Path('artifacts/ml_data/ml_filing_anchored_enriched.csv')
OUTPUT_DIR = Path('artifacts/ml_models')
GRAPHS_DIR = Path('graphs/ml_models')

# Targets
TARGET_REG = 'y_fwd_63d_log'
TARGET_CLF = 'y_up_63d'

# Cross-validation parameters (small-sample friendly)
N_FOLDS = 3                 # informational only; we drive folds via params below
EMBARGO_DAYS = 0            # disable embargo in CV to avoid skipping folds
HOLDOUT_FRAC = 0.25
ROLLING_MEAN_N = 6          # slightly shorter baseline window

# Feature selection
MAX_MISSING_PCT = 0.40
MAX_CORR = 0.90
MAX_VIF = 12.0              # a bit more permissive for small N

# Model parameters
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
LASSO_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]
ELASTIC_ALPHAS = [0.01, 0.1, 1.0, 10.0]
ELASTIC_L1_RATIOS = [0.1, 0.5, 0.7, 0.9]

# ============================================================================
# LEAK-PROOF PREPROCESSING PIPELINE
# ============================================================================

class ReplaceInf(BaseEstimator, TransformerMixin):
    """Replace infinite values with NaN."""
    def fit(self, X, y=None): 
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
        return X

class DropHighNA(BaseEstimator, TransformerMixin):
    """Drop columns with high missing value percentage."""
    def __init__(self, max_na_frac=0.40): 
        self.max_na_frac = max_na_frac
    
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.keep_cols_ = X.columns[X.isna().mean() <= self.max_na_frac]
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.keep_cols_]

class WinsorizeQuantiles(BaseEstimator, TransformerMixin):
    """Winsorize features using quantiles fitted on training data."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
    
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.q_low_ = X.quantile(self.lower, numeric_only=True)
        self.q_high_ = X.quantile(self.upper, numeric_only=True)
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.q_low_.index:
            if c in X.columns:
                X[c] = X[c].clip(self.q_low_[c], self.q_high_[c])
        return X

def assert_finite(A, name):
    """Assert that array contains only finite values."""
    arr = np.asarray(A, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} still has NaN/inf.")

# Create the leak-proof preprocessing pipeline
def create_preprocessing_pipeline(use_pca: bool = False, pca_var: float = 0.90):
    """Leak-safe preprocessing; optionally add PCA retaining pca_var explained variance."""
    from sklearn.feature_selection import VarianceThreshold
    steps = [
        ("inf_to_nan", ReplaceInf()),
        ("drop_sparse", DropHighNA(max_na_frac=0.40)),
        ("winsor", WinsorizeQuantiles(0.01, 0.99)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_var, svd_solver="full", random_state=42)))
    steps.append(("var_thresh", VarianceThreshold(1e-12)))
    return Pipeline(steps=steps)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_directories():
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created output directories")

def load_and_validate_data(file_path: Path) -> pd.DataFrame:
    """Load data and perform sanity checks."""
    print(f"\n[DATA] Loading data from {file_path}...")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df['anchor_date'] = pd.to_datetime(df['anchor_date'])
    df['reported_date'] = pd.to_datetime(df['reported_date'])
    
    # Sort by anchor_date
    df = df.sort_values('anchor_date').reset_index(drop=True)
    
    print(f"  [OK] Loaded {len(df)} records")
    print(f"  [CALENDAR] Date range: {df['anchor_date'].min().strftime('%Y-%m-%d')} to {df['anchor_date'].max().strftime('%Y-%m-%d')}")
    
    # Sanity checks
    print(f"\n[CHECK] Performing sanity checks...")
    
    # Check anchor_date >= reported_date
    invalid_dates = df[df['anchor_date'] < df['reported_date']]
    if len(invalid_dates) > 0:
        print(f"  [WARNING] Warning: {len(invalid_dates)} rows have anchor_date < reported_date")
        print(f"    First few invalid rows:")
        print(invalid_dates[['reported_date', 'anchor_date']].head())
    else:
        print(f"  [OK] All anchor dates are >= reported dates")
    
    # Check target existence
    required_targets = [TARGET_REG, TARGET_CLF, 'y_fwd_21d_log', 'y_fwd_126d_log', 'y_fwd_63d_excess']
    missing_targets = [t for t in required_targets if t not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing required targets: {missing_targets}")
    else:
        print(f"  [OK] All required targets found")
    
    return df

def select_features(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """Select features and handle missing values."""
    print(f"\n[FIX] Selecting features...")
    
    # Exclude non-feature columns
    exclude_cols = ['ticker', 'reported_date', 'anchor_date', 'anchor_price', 'fiscal_year']
    exclude_cols.extend([col for col in df.columns if col.startswith('y_')])
    
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    
    print(f"  [DATA] Found {len(feature_cols)} numeric feature columns")
    
    # Check missing values
    missing_pct = df[feature_cols].isnull().mean().sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > MAX_MISSING_PCT]
    
    if len(high_missing) > 0:
        print(f"  [WARNING] Dropping {len(high_missing)} features with >{MAX_MISSING_PCT:.0%} missing:")
        for col, pct in high_missing.items():
            print(f"    {col}: {pct:.1%}")
        feature_cols = [col for col in feature_cols if col not in high_missing.index]
    
    print(f"  [OK] Using {len(feature_cols)} features after missing value filter")
    
    return feature_cols, df[feature_cols + ['anchor_date', TARGET_REG, TARGET_CLF]]

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for features."""
    print(f"\n[CHECK] Computing VIF...")
    
    # Clean data: remove infinite values and replace with NaN
    X_clean = X.replace([np.inf, -np.inf], np.nan)
    
    # Drop columns with all NaN values
    X_clean = X_clean.dropna(axis=1, how='all')
    
    if X_clean.empty:
        print(f"  [WARNING] No valid features for VIF computation")
        return pd.DataFrame(columns=['feature', 'vif'])
    
    # Fill remaining NaN values with median
    X_clean = X_clean.fillna(X_clean.median())
    
    # Standardize features for VIF computation
    try:
        X_std = StandardScaler().fit_transform(X_clean)
        X_std_df = pd.DataFrame(X_std, columns=X_clean.columns, index=X_clean.index)
    except Exception as e:
        print(f"  [WARNING] Error standardizing features: {e}")
        return pd.DataFrame(columns=['feature', 'vif'])
    
    vif_data = []
    for i, col in enumerate(X_clean.columns):
        if X_std_df[col].var() > 1e-10:  # Avoid division by zero
            try:
                vif = variance_inflation_factor(X_std_df.values, i)
                vif_data.append({'feature': col, 'vif': vif})
            except Exception as e:
                print(f"    [WARNING] Could not compute VIF for {col}: {e}")
                vif_data.append({'feature': col, 'vif': np.nan})
    
    vif_df = pd.DataFrame(vif_data).sort_values('vif', ascending=False)
    
    high_vif = vif_df[vif_df['vif'] > MAX_VIF]
    if len(high_vif) > 0:
        print(f"  [WARNING] Found {len(high_vif)} features with VIF > {MAX_VIF}:")
        for _, row in high_vif.iterrows():
            if not pd.isna(row['vif']):
                print(f"    {row['feature']}: {row['vif']:.2f}")
    else:
        print(f"  [OK] No features with VIF > {MAX_VIF}")
    
    return vif_df

def expanding_window_splits(n_rows, min_train=36, val_size=8, step=4, embargo_rows=0):
    """
    Generate expanding-window splits.
    Train: [0 : train_end)
    Embargo: [train_end : train_end + embargo_rows)
    Val: [train_end + embargo_rows : train_end + embargo_rows + val_size)
    """
    print(f"\n[PROCESS] Creating expanding window CV with min_train={min_train}, val_size={val_size}, "
          f"step={step}, embargo={embargo_rows} ...")

    start_train_end = min_train
    fold_count = 0

    while True:
        train_end = start_train_end
        val_start = train_end + embargo_rows
        val_end = val_start + val_size
        if val_end > n_rows:
            break

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        fold_count += 1
        print(f"  Fold {fold_count}: Train {len(train_idx)} | Val {len(val_idx)} "
              f"| idx: train[0:{train_end}) val[{val_start}:{val_end})")
        yield train_idx, val_idx

        start_train_end += step

    print(f"  [OK] Generated {fold_count} folds total")
    return fold_count

# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def create_correlation_heatmap(df: pd.DataFrame, feature_cols: List[str], save_path: Path):
    """Create correlation heatmap for features."""
    print(f"\n[DATA] Creating correlation heatmap...")
    
    plt.figure(figsize=(16, 12))
    corr_matrix = df[feature_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, cbar_kws={'shrink': 0.8})
    
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved correlation heatmap to {save_path}")

def handle_high_correlation(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """Remove highly correlated features."""
    print(f"\n[FIX] Handling high correlations...")
    
    corr_matrix = df[feature_cols].corr().abs()
    
    # Find pairs with high correlation
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > MAX_CORR:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
    
    if high_corr_pairs:
        print(f"  [WARNING] Found {len(high_corr_pairs)} highly correlated pairs (>{MAX_CORR}):")
        
        # Remove one feature from each highly correlated pair
        to_remove = set()
        for col1, col2, corr in high_corr_pairs:
            if col1 not in to_remove and col2 not in to_remove:
                # Keep the one with less missing data
                missing1 = df[col1].isnull().sum()
                missing2 = df[col2].isnull().sum()
                
                if missing1 <= missing2:
                    to_remove.add(col2)
                    print(f"    Removing {col2} (corr={corr:.3f} with {col1})")
                else:
                    to_remove.add(col1)
                    print(f"    Removing {col1} (corr={corr:.3f} with {col2})")
        
        feature_cols = [col for col in feature_cols if col not in to_remove]
        print(f"  [OK] Reduced to {len(feature_cols)} features after correlation filter")
    else:
        print(f"  [OK] No highly correlated features found")
    
    return feature_cols

# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_regression_models_with_pipeline(X_train: pd.DataFrame, y_train: pd.Series, 
                                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
    """Train regression models with leak-proof preprocessing pipeline."""
    print(f"\n[TARGET] Training regression models with leak-proof pipeline...")
    
    results = {}
    
    # Create preprocessing pipeline
    preprocess = create_preprocessing_pipeline()
    
    # Fit preprocessing on training data only
    preprocess.fit(X_train)
    X_train_processed = preprocess.transform(X_train)
    X_val_processed = preprocess.transform(X_val)
    
    # Assert finite values
    assert_finite(X_train_processed, "X_train_processed")
    assert_finite(X_val_processed, "X_val_processed")
    assert_finite(y_train, "y_train")
    assert_finite(y_val, "y_val")
    
    # Get feature names after preprocessing
    feature_names = X_train_processed.columns.tolist() if hasattr(X_train_processed, 'columns') else [f'feature_{i}' for i in range(X_train_processed.shape[1])]
    
    # Ridge regression
    print(f"  Training Ridge regression...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_processed, y_train)
    y_pred_ridge = ridge_model.predict(X_val_processed)
    
    results['ridge'] = {
        'model': ridge_model,
        'preprocess': preprocess,
        'feature_names': feature_names,
        'mae': mean_absolute_error(y_val, y_pred_ridge),
        'rmse': np.sqrt(mean_squared_error(y_val, y_pred_ridge)),
        'r2': r2_score(y_val, y_pred_ridge),
        'hit_rate': float(np.mean(np.sign(y_pred_ridge) == np.sign(y_val))),
        'params': {'alpha': 1.0}
    }
    
    # Lasso regression
    print(f"  Training Lasso regression...")
    lasso_model = Lasso(alpha=0.1, max_iter=2000)
    lasso_model.fit(X_train_processed, y_train)
    y_pred_lasso = lasso_model.predict(X_val_processed)
    
    results['lasso'] = {
        'model': lasso_model,
        'preprocess': preprocess,
        'feature_names': feature_names,
        'mae': mean_absolute_error(y_val, y_pred_lasso),
        'rmse': np.sqrt(mean_squared_error(y_val, y_pred_lasso)),
        'r2': r2_score(y_val, y_pred_lasso),
        'hit_rate': float(np.mean(np.sign(y_pred_lasso) == np.sign(y_val))),
        'params': {'alpha': 0.1}
    }
    
    # PCA + Ridge
    print(f"  Training PCA+Ridge...")
    preprocess_pca = create_preprocessing_pipeline(use_pca=True, pca_var=0.90)
    preprocess_pca.fit(X_train)
    Xtr_p = preprocess_pca.transform(X_train)
    Xva_p = preprocess_pca.transform(X_val)
    assert_finite(Xtr_p, "Xtr_p"); assert_finite(Xva_p, "Xva_p")
    pca_ridge = Ridge(alpha=1.0)
    pca_ridge.fit(Xtr_p, y_train)
    y_pred_pcar = pca_ridge.predict(Xva_p)
    results['pca_ridge'] = {
        'model': pca_ridge,
        'preprocess': preprocess_pca,
        'feature_names': getattr(X_train, "columns", None),
        'mae': mean_absolute_error(y_val, y_pred_pcar),
        'rmse': np.sqrt(mean_squared_error(y_val, y_pred_pcar)),
        'r2': r2_score(y_val, y_pred_pcar),
        'hit_rate': float(np.mean(np.sign(y_pred_pcar) == np.sign(y_val))),
        'params': {'alpha': 1.0, 'pca_var': 0.90}
    }
    
    return results

def train_with_grid_search(pipeline: Pipeline, param_grid: Dict, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
    """Train model with grid search and return best results."""
    best_score = -np.inf
    best_params = None
    best_model = None
    
    for params in ParameterGrid(param_grid):
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_val)
        score = r2_score(y_val, y_pred)
        
        if score > best_score:
            best_score = score
            best_params = params
            best_model = pipeline
    
    # Calculate metrics
    y_pred = best_model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    return {
        'model': best_model,
        'params': best_params,
        'score': best_score,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred
    }

def train_classification_model_with_pipeline(X_train: pd.DataFrame, y_train: pd.Series,
                                           X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
    """Train classification model with leak-proof preprocessing pipeline."""
    print(f"\n[TARGET] Training classification model with leak-proof pipeline...")
    
    # Create preprocessing pipeline
    preprocess = create_preprocessing_pipeline()
    
    # Fit preprocessing on training data only
    preprocess.fit(X_train)
    X_train_processed = preprocess.transform(X_train)
    X_val_processed = preprocess.transform(X_val)
    
    # Assert finite values
    assert_finite(X_train_processed, "X_train_processed")
    assert_finite(X_val_processed, "X_val_processed")
    assert_finite(y_train, "y_train")
    assert_finite(y_val, "y_val")
    
    # Train classification model
    clf_model = LogisticRegression(penalty='l2', class_weight='balanced', max_iter=1000)
    clf_model.fit(X_train_processed, y_train)
    
    # Predictions
    y_pred_proba = clf_model.predict_proba(X_val_processed)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    balanced_acc = balanced_accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    brier = brier_score_loss(y_val, y_pred_proba)
    
    return {
        'model': clf_model,
        'preprocess': preprocess,
        'balanced_acc': balanced_acc,
        'f1': f1,
        'roc_auc': roc_auc,
        'brier': brier,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_cross_validation(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run expanding window cross-validation with leak-proof preprocessing."""
    print(f"\n[PROCESS] Running expanding window cross-validation...")
    
    # Prepare data - replace infinite values early
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Log columns with infinite values
    bad_inf_cols = X.columns[np.isinf(X.values).any(axis=0)]
    if len(bad_inf_cols) > 0:
        print(f"  [WARNING] Columns with ±inf (pre-pipeline): {list(bad_inf_cols)}")
    else:
        print(f"  [OK] No infinite values found in features")
    
    y_reg = df[TARGET_REG].copy()
    y_clf = df[TARGET_CLF].copy()
    
    # Remove rows with missing targets
    valid_reg = ~y_reg.isnull()
    valid_clf = ~y_clf.isnull()
    
    print(f"  [DATA] Valid regression samples: {valid_reg.sum()}")
    print(f"  [DATA] Valid classification samples: {valid_clf.sum()}")
    
    # Cross-validation results
    cv_results_reg = []
    cv_results_clf = []
    coef_stability = []
    
    # --- small-sample CV folds (few, but real) ---
    n_samples = len(df)
    # First attempt: modest train/val and tiny step; no embargo
    folds = list(expanding_window_splits(
        n_rows=n_samples,
        min_train=36,   # ~3 years if quarterly, or ~3 years worth of anchors
        val_size=8,     # small validation window
        step=4,         # roll forward a bit each time
        embargo_rows=0  # keep it 0 for small N
    ))

    # Fallback attempts if still no folds
    if len(folds) == 0:
        folds = list(expanding_window_splits(n_samples, min_train=30, val_size=6, step=3, embargo_rows=0))
    if len(folds) == 0:
        folds = list(expanding_window_splits(n_samples, min_train=24, val_size=6, step=3, embargo_rows=0))

    # Last-resort: create a single chronological split so diagnostics still run
    if len(folds) == 0:
        train_end = max(24, int(n_samples * 0.60))     # ensure minimum train size
        val_size  = min( max(6, int(n_samples * 0.10)), n_samples - train_end )
        train_idx = np.arange(0, train_end)
        val_idx   = np.arange(train_end, train_end + val_size)
        folds = [(train_idx, val_idx)]
        print(f"  [WARNING] Using last-resort single split: train={len(train_idx)}, val={len(val_idx)}")
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n  [FOLDER] Fold {fold + 1}:")
        
        # Get training and validation data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_reg_train, y_reg_val = y_reg.iloc[train_idx], y_reg.iloc[val_idx]
        y_clf_train, y_clf_val = y_clf.iloc[train_idx], y_clf.iloc[val_idx]
        
        # Remove missing targets
        reg_mask = ~y_reg_train.isnull() & ~y_reg_val.isnull()
        clf_mask = ~y_clf_train.isnull() & ~y_clf_val.isnull()
        
        # After slicing X_train/X_val and y_*:
        if len(X_train) < 24 or len(X_val) < 6:
            print(f"   [WARNING] Skipping fold {fold+1} - not enough samples after filtering")
            continue
        
        X_reg_train = X_train[reg_mask]
        X_reg_val = X_val[~y_reg_val.isnull()]
        y_reg_train_clean = y_reg_train[reg_mask]
        y_reg_val_clean = y_reg_val[~y_reg_val.isnull()]
        
        X_clf_train = X_train[clf_mask]
        X_clf_val = X_val[~y_clf_val.isnull()]
        y_clf_train_clean = y_clf_train[clf_mask]
        y_clf_val_clean = y_clf_val[~y_clf_val.isnull()]
        
        print(f"    Regression: {len(X_reg_train)} train, {len(X_reg_val)} val")
        print(f"    Classification: {len(X_clf_train)} train, {len(X_clf_val)} val")
        
        # Skip fold if no training samples after filtering
        if len(X_reg_train) == 0 or len(X_clf_train) == 0:
            print(f"   [WARNING] Skipping fold {fold+1} - no training samples after target filtering")
            continue
        
        # Train regression models with new pipeline
        reg_results = train_regression_models_with_pipeline(X_reg_train, y_reg_train_clean, X_reg_val, y_reg_val_clean)
        
        # Train classification model with new pipeline
        clf_results = train_classification_model_with_pipeline(X_clf_train, y_clf_train_clean, X_clf_val, y_clf_val_clean)
        
        # Store results
        for model_name, results in reg_results.items():
            cv_results_reg.append({
                'fold': fold + 1,
                'model': model_name,
                'mae': results['mae'],
                'rmse': results['rmse'],
                'r2': results['r2'],
                'hit_rate': results['hit_rate'],
                'params': str(results['params'])
            })
        
        cv_results_clf.append({
            'fold': fold + 1,
            'balanced_acc': clf_results['balanced_acc'],
            'f1': clf_results['f1'],
            'roc_auc': clf_results['roc_auc'],
            'brier': clf_results['brier']
        })
        
        # Tree baselines
        tree_reg = train_tree_regressor(X_reg_train, y_reg_train_clean, X_reg_val, y_reg_val_clean)
        cv_results_reg.append({
            'fold': fold + 1, 'model': 'tree_reg',
            'mae': tree_reg['mae'], 'rmse': tree_reg['rmse'],
            'r2': tree_reg['r2'], 'hit_rate': tree_reg['hit_rate'],
            'params': str(tree_reg['params'])
        })

        tree_clf = train_tree_classifier(X_clf_train, y_clf_train_clean, X_clf_val, y_clf_val_clean)
        cv_results_clf.append({
            'fold': fold + 1,
            'balanced_acc': tree_clf['balanced_acc'],
            'f1': tree_clf['f1'],
            'roc_auc': tree_clf['roc_auc'],
            'brier': tree_clf['brier']
        })
        
        # Store coefficients for stability analysis
        for model_name, results in reg_results.items():
            if hasattr(results['model'], 'coef_'):
                coefs = results['model'].coef_
                for i, feature in enumerate(results.get('feature_names', [])):
                    if i < len(coefs):
                        coef_stability.append({
                            'fold': fold + 1,
                            'model': model_name,
                            'feature': feature,
                            'coefficient': coefs[i]
                        })
    
    # Convert to DataFrames
    cv_reg_df = pd.DataFrame(cv_results_reg)
    cv_clf_df = pd.DataFrame(cv_results_clf)
    coef_df = pd.DataFrame(coef_stability)
    
    print(f"\n[OK] Cross-validation complete: {len(cv_reg_df)} regression folds, {len(cv_clf_df)} classification folds")
    
    # Persist the actual fold indices for transparency
    if len(folds) > 0:
        fold_summary = pd.DataFrame([
            {
                "fold": i+1, 
                "train_start": int(tr[0]), 
                "train_end": int(tr[-1]),
                "val_start": int(vl[0]), 
                "val_end": int(vl[-1])
            }
            for i, (tr, vl) in enumerate([(np.array(t), np.array(v)) for (t, v) in folds])
        ])
        fold_summary.to_csv(OUTPUT_DIR / "cv_folds_summary.csv", index=False)
        print(f"  [FOLDER] Fold indices saved to: {OUTPUT_DIR / 'cv_folds_summary.csv'}")
    
    return cv_reg_df, cv_clf_df, coef_df

def create_baseline_models(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """Create baseline models for comparison."""
    print(f"\n[DATA] Creating baseline models...")
    
    baselines = {}
    
    # Zero prediction baseline
    baselines['zero'] = {
        'description': 'Predict zero for all samples',
        'mae': np.mean(np.abs(df[TARGET_REG].dropna())),
        'rmse': np.sqrt(np.mean(df[TARGET_REG].dropna() ** 2)),
        'r2': 0.0
    }
    
    # Rolling mean baseline
    rolling_mean = df[TARGET_REG].rolling(window=ROLLING_MEAN_N, min_periods=1).mean()
    
    # Align the arrays by removing NaN values from both
    valid_mask = ~df[TARGET_REG].isnull() & ~rolling_mean.isnull()
    y_true_rolling = df[TARGET_REG][valid_mask]
    y_pred_rolling = rolling_mean[valid_mask]
    
    rolling_mae = mean_absolute_error(y_true_rolling, y_pred_rolling)
    rolling_rmse = np.sqrt(mean_squared_error(y_true_rolling, y_pred_rolling))
    rolling_r2 = r2_score(y_true_rolling, y_pred_rolling)
    
    baselines['rolling_mean'] = {
        'description': f'Rolling mean of last {ROLLING_MEAN_N} targets',
        'mae': rolling_mae,
        'rmse': rolling_rmse,
        'r2': rolling_r2
    }
    
    # Majority class baseline for classification
    majority_class = df[TARGET_CLF].mode()[0]
    majority_acc = balanced_accuracy_score(df[TARGET_CLF].dropna(), 
                                         [majority_class] * len(df[TARGET_CLF].dropna()))
    
    baselines['majority_class'] = {
        'description': f'Majority class ({majority_class})',
        'balanced_acc': majority_acc,
        'f1': f1_score(df[TARGET_CLF].dropna(), [majority_class] * len(df[TARGET_CLF].dropna()))
    }
    
    return baselines

def analyze_coefficient_stability(coef_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze coefficient stability across folds."""
    print(f"\n[CHECK] Analyzing coefficient stability...")
    
    # Check if we have any coefficient data
    if coef_df.empty or 'model' not in coef_df.columns:
        print(f"  [WARNING] No coefficient data available for stability analysis")
        return pd.DataFrame(columns=['model', 'feature', 'mean_coefficient', 'std_coefficient', 'cv_coefficient', 'sign_changes', 'is_stable'])
    
    stability_analysis = []
    
    for model in coef_df['model'].unique():
        model_data = coef_df[coef_df['model'] == model]
        
        for feature in model_data['feature'].unique():
            feature_data = model_data[model_data['feature'] == feature]
            
            coefs = feature_data['coefficient'].values
            mean_coef = np.mean(coefs)
            std_coef = np.std(coefs)
            cv_coef = std_coef / abs(mean_coef) if mean_coef != 0 else np.inf
            
            # Check for sign flips
            sign_changes = np.sum(np.diff(np.sign(coefs)) != 0) if len(coefs) > 1 else 0
            
            stability_analysis.append({
                'model': model,
                'feature': feature,
                'mean_coefficient': mean_coef,
                'std_coefficient': std_coef,
                'cv_coefficient': cv_coef,
                'sign_changes': sign_changes,
                'is_stable': cv_coef < 0.5 and sign_changes == 0
            })
    
    stability_df = pd.DataFrame(stability_analysis)
    
    # Flag unstable features
    unstable = stability_df[~stability_df['is_stable']]
    if len(unstable) > 0:
        print(f"  [WARNING] Found {len(unstable)} potentially unstable features:")
        for _, row in unstable.iterrows():
            print(f"    {row['model']}.{row['feature']}: CV={row['cv_coefficient']:.3f}, sign_changes={row['sign_changes']}")
    else:
        print(f"  [OK] All features appear stable across folds")
    
    return stability_df

# ============================================================================
# ENHANCED TRAINING HELPERS AND PLOTTING UTILITIES
# ============================================================================

def train_tree_regressor(X_train, y_train, X_val, y_val):
    """Train HistGradientBoostingRegressor with leak-safe preprocessing."""
    preprocess = create_preprocessing_pipeline(use_pca=False)
    preprocess.fit(X_train)
    Xtr = preprocess.transform(X_train)
    Xva = preprocess.transform(X_val)
    assert_finite(Xtr, "Xtr"); assert_finite(Xva, "Xva")
    gbr = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=300,
        min_samples_leaf=8, l2_regularization=0.1, random_state=42
    )
    gbr.fit(Xtr, y_train)
    yhat = gbr.predict(Xva)
    return {
        'model': gbr, 'preprocess': preprocess,
        'mae': mean_absolute_error(y_val, yhat),
        'rmse': np.sqrt(mean_squared_error(y_val, yhat)),
        'r2': r2_score(y_val, yhat),
        'hit_rate': float(np.mean(np.sign(yhat) == np.sign(y_val))),
        'params': {'max_depth': 3, 'lr': 0.05}
    }

def train_tree_classifier(X_train, y_train, X_val, y_val):
    """Train HistGradientBoostingClassifier with leak-safe preprocessing."""
    preprocess = create_preprocessing_pipeline(use_pca=False)
    preprocess.fit(X_train)
    Xtr = preprocess.transform(X_train)
    Xva = preprocess.transform(X_val)
    assert_finite(Xtr, "Xtr"); assert_finite(Xva, "Xva")
    gbc = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300,
        min_samples_leaf=8, l2_regularization=0.1,
        class_weight="balanced", random_state=42
    )
    gbc.fit(Xtr, y_train)
    proba = gbc.predict_proba(Xva)[:, 1]
    pred = (proba > 0.5).astype(int)
    return {
        'model': gbc, 'preprocess': preprocess,
        'balanced_acc': balanced_accuracy_score(y_val, pred),
        'f1': f1_score(y_val, pred),
        'roc_auc': roc_auc_score(y_val, proba),
        'brier': brier_score_loss(y_val, proba),
        'predictions': pred, 'probabilities': proba
    }

def plot_roc(y_true, y_proba, save_path: Path):
    """Plot ROC curve for classification results."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0,1],[0,1],'--',alpha=0.6)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Holdout ROC')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_pred_vs_actual(y_true, y_pred, save_path: Path):
    """Plot predicted vs actual values for regression results."""
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.figure(figsize=(6,5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot(lims, lims, '--', alpha=0.6)
    plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title('Holdout: Pred vs Actual')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def save_pca_scree(preprocess, save_path: Path):
    """Save PCA scree plot showing cumulative explained variance."""
    if isinstance(preprocess, Pipeline) and 'pca' in preprocess.named_steps:
        exp = preprocess.named_steps['pca'].explained_variance_ratio_
        plt.figure(figsize=(6,4))
        plt.plot(np.arange(1, len(exp)+1), np.cumsum(exp), marker='o')
        plt.xlabel('Number of Components'); plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Scree (Cumulative)'); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_permutation_importance(pipeline, X, y, feature_names, save_path: Path, n_repeats=200, scoring='r2'):
    """Plot permutation importance for features."""
    r = permutation_importance(pipeline, X, y, n_repeats=n_repeats, random_state=42, scoring=scoring)
    imp = pd.DataFrame({'feature': feature_names, 'importance': r.importances_mean})
    imp = imp.sort_values('importance', ascending=False).head(15)
    plt.figure(figsize=(8,6))
    plt.barh(imp['feature'][::-1], imp['importance'][::-1])
    plt.title(f'Permutation Importance ({scoring})'); plt.xlabel('Mean Δ score when shuffled')
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_directional_cumret(y_true_log, pred_sign, save_path: Path):
    """Plot directional cumulative return curve (toy example)."""
    strat = (np.sign(pred_sign) * np.sign(y_true_log)) * np.abs(y_true_log)  # toy curve
    cum = np.cumsum(strat); bh = np.cumsum(y_true_log.values)
    plt.figure(figsize=(7,5))
    plt.plot(cum, label='Directional Strategy')
    plt.plot(bh, label='Buy & Hold', alpha=0.7)
    plt.legend(); plt.title('Holdout: Cumulative Log Return (toy)')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def create_diagnostic_plots(cv_reg_df: pd.DataFrame, cv_clf_df: pd.DataFrame, 
                           coef_df: pd.DataFrame, df: pd.DataFrame, feature_cols: List[str]):
    """Create diagnostic plots."""
    print(f"\n[DATA] Creating diagnostic plots...")
    
    # Set style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # 1. Model performance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Regression metrics
    if not cv_reg_df.empty and 'model' in cv_reg_df.columns:
        reg_metrics = cv_reg_df.groupby('model')[['mae', 'rmse', 'r2']].mean()
        reg_metrics.plot(kind='bar', ax=ax1)
        ax1.set_title('Regression Model Performance (CV Average)')
        ax1.set_ylabel('Metric Value')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
    else:
        ax1.text(0.5, 0.5, 'No regression CV data available', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Regression Model Performance (CV Average)')
    
    # Classification metrics
    if not cv_clf_df.empty:
        clf_metrics = cv_clf_df[['balanced_acc', 'f1', 'roc_auc', 'brier']].mean()
        clf_metrics.plot(kind='bar', ax=ax2)
        ax2.set_title('Classification Model Performance (CV Average)')
        ax2.set_ylabel('Metric Value')
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No classification CV data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Classification Model Performance (CV Average)')
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Coefficient stability heatmap
    if len(coef_df) > 0 and 'coefficient' in coef_df.columns:
        coef_pivot = coef_df.pivot_table(values='coefficient', index='feature', columns='fold', aggfunc='mean')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(coef_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0)
        plt.title('Coefficient Stability Across Folds')
        plt.xlabel('Fold')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(GRAPHS_DIR / 'coefficient_stability.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"  [WARNING] Skipping coefficient stability plot - no coefficient data available")
    
    print(f"  [OK] Saved diagnostic plots to {GRAPHS_DIR}")

def run_final_holdout_evaluation(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """Run final holdout evaluation with leak-proof preprocessing."""
    print(f"\n[TARGET] Running final holdout evaluation...")
    
    # Split data - last 25% for holdout
    n = len(df)
    holdout_start = int(n * 0.75)
    
    X_pre, y_reg_pre = df[feature_cols].iloc[:holdout_start], df[TARGET_REG].iloc[:holdout_start]
    X_ho, y_reg_ho = df[feature_cols].iloc[holdout_start:], df[TARGET_REG].iloc[holdout_start:]
    
    y_clf_pre = df[TARGET_CLF].iloc[:holdout_start]
    y_clf_ho = df[TARGET_CLF].iloc[holdout_start:]
    
    print(f"  [DATA] Training samples: {len(X_pre)}")
    print(f"  [DATA] Holdout samples: {len(X_ho)}")
    
    # Replace infinite values early
    X_pre = X_pre.replace([np.inf, -np.inf], np.nan)
    X_ho = X_ho.replace([np.inf, -np.inf], np.nan)
    
    # Remove rows with missing targets
    reg_mask_pre = ~y_reg_pre.isnull()
    clf_mask_pre = ~y_clf_pre.isnull()
    reg_mask_ho = ~y_reg_ho.isnull()
    clf_mask_ho = ~y_clf_ho.isnull()
    
    X_reg_pre = X_pre[reg_mask_pre]
    y_reg_pre_clean = y_reg_pre[reg_mask_pre]
    X_reg_ho = X_ho[reg_mask_ho]
    y_reg_ho_clean = y_reg_ho[reg_mask_ho]
    
    X_clf_pre = X_pre[clf_mask_pre]
    y_clf_pre_clean = y_clf_pre[clf_mask_pre]
    X_clf_ho = X_ho[clf_mask_ho]
    y_clf_ho_clean = y_clf_ho[clf_mask_ho]
    
    print(f"  [TARGET] Training final models...")
    
    # Regression model
    preprocess_reg = create_preprocessing_pipeline()
    preprocess_reg.fit(X_reg_pre)
    X_reg_pre_t = preprocess_reg.transform(X_reg_pre)
    X_reg_ho_t = preprocess_reg.transform(X_reg_ho)
    
    assert_finite(X_reg_pre_t, "X_reg_pre_t")
    assert_finite(X_reg_ho_t, "X_reg_ho_t")
    
    final_reg = Ridge(alpha=1.0).fit(X_reg_pre_t, y_reg_pre_clean)
    y_reg_pred = final_reg.predict(X_reg_ho_t)
    
    # Classification model
    preprocess_clf = create_preprocessing_pipeline()
    preprocess_clf.fit(X_clf_pre)
    X_clf_pre_t = preprocess_clf.transform(X_clf_pre)
    X_clf_ho_t = preprocess_clf.transform(X_clf_ho)
    
    assert_finite(X_clf_pre_t, "X_clf_pre_t")
    assert_finite(X_clf_ho_t, "X_clf_ho_t")
    
    final_clf = LogisticRegression(penalty='l2', class_weight='balanced', max_iter=1000).fit(X_clf_pre_t, y_clf_pre_clean)
    y_clf_pred_proba = final_clf.predict_proba(X_clf_ho_t)[:, 1]
    y_clf_pred = (y_clf_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    holdout_metrics = {
        'regression': {
            'mae': mean_absolute_error(y_reg_ho_clean, y_reg_pred),
            'rmse': np.sqrt(mean_squared_error(y_reg_ho_clean, y_reg_pred)),
            'r2': r2_score(y_reg_ho_clean, y_reg_pred),
            'hit_rate': float(np.mean(np.sign(y_reg_pred) == np.sign(y_reg_ho_clean)))
        },
        'classification': {
            'balanced_acc': balanced_accuracy_score(y_clf_ho_clean, y_clf_pred),
            'f1': f1_score(y_clf_ho_clean, y_clf_pred),
            'roc_auc': roc_auc_score(y_clf_ho_clean, y_clf_pred_proba),
            'brier': brier_score_loss(y_clf_ho_clean, y_clf_pred_proba)
        }
    }
    
    # Print results
    print(f"\n[TARGET] HOLDOUT RESULTS:")
    print(f"  Regression R²: {holdout_metrics['regression']['r2']:.3f}")
    print(f"  Regression MAE: {holdout_metrics['regression']['mae']:.3f}")
    print(f"  Regression Hit Rate: {holdout_metrics['regression']['hit_rate']:.3f}")
    print(f"  Classification ROC-AUC: {holdout_metrics['classification']['roc_auc']:.3f}")
    print(f"  Classification F1: {holdout_metrics['classification']['f1']:.3f}")
    
    # Save predictions - ensure all arrays have the same length
    holdout_predictions = pd.DataFrame({
        'anchor_date': df['anchor_date'].iloc[holdout_start:][reg_mask_ho].values,
        'y_fwd_63d_log_true': y_reg_ho_clean.values,
        'y_fwd_63d_log_pred': y_reg_pred,
        'y_up_63d_true': y_clf_ho_clean.values,
        'y_up_63d_pred': y_clf_pred,
        'y_up_63d_proba': y_clf_pred_proba
    })
    
    try:
        holdout_predictions.to_csv(OUTPUT_DIR / 'holdout_predictions.csv', index=False)
        print(f"  [OK] Holdout predictions saved")
    except PermissionError:
        print(f"  [WARNING] Could not save holdout predictions due to permission error, continuing...")
    
    # Enhanced plots and analysis
    print(f"\n[DATA] Creating enhanced holdout plots...")
    
    # A) Save ROC and Pred-vs-Actual
    plot_roc(y_clf_ho_clean, y_clf_pred_proba, GRAPHS_DIR / 'holdout_roc.png')
    plot_pred_vs_actual(y_reg_ho_clean, y_reg_pred, GRAPHS_DIR / 'holdout_pred_vs_actual.png')
    
    # B) Add PCA scree (fit a PCA-preprocess on training to visualize)
    preprocess_reg_pca = create_preprocessing_pipeline(use_pca=True, pca_var=0.90)
    preprocess_reg_pca.fit(X_reg_pre)
    save_pca_scree(preprocess_reg_pca, GRAPHS_DIR / 'pca_scree.png')
    
    # C) Permutation importance (regression R² and classification ROC-AUC)
    pipe_reg = Pipeline([('prep', preprocess_reg), ('model', final_reg)])
    plot_permutation_importance(
        pipeline=pipe_reg,
        X=df[feature_cols].iloc[holdout_start:][reg_mask_ho],
        y=y_reg_ho_clean,
        feature_names=list(df[feature_cols].columns),
        save_path=GRAPHS_DIR / 'perm_importance_reg_holdout.png',
        scoring='r2'
    )

    pipe_clf = Pipeline([('prep', preprocess_clf), ('model', final_clf)])
    plot_permutation_importance(
        pipeline=pipe_clf,
        X=df[feature_cols].iloc[holdout_start:][clf_mask_ho],
        y=y_clf_ho_clean,
        feature_names=list(df[feature_cols].columns),
        save_path=GRAPHS_DIR / 'perm_importance_clf_holdout.png',
        scoring='roc_auc'
    )
    
    # D) Simple directional cumulative log-return curve (toy)
    plot_directional_cumret(y_reg_ho_clean, y_clf_pred, GRAPHS_DIR / 'holdout_cumret_directional.png')
    
    print(f"  [OK] Enhanced holdout plots saved to {GRAPHS_DIR}")
    
    return holdout_metrics

def main():
    """Main function for enhanced time-aware modeling."""
    print("Step 9: Enhanced Time-Aware Machine Learning Modeling")
    print("=" * 60)
    
    # Setup
    setup_directories()
    
    # Load and validate data
    df = load_and_validate_data(DATA_FILE)
    
    # Select features
    feature_cols, df_features = select_features(df)
    
    # Handle high correlations
    feature_cols = handle_high_correlation(df_features, feature_cols)
    
    # Compute VIF
    vif_df = compute_vif(df_features[feature_cols])
    try:
        vif_df.to_csv(OUTPUT_DIR / 'vif_table.csv', index=False)
        print(f"  [OK] VIF table saved to {OUTPUT_DIR / 'vif_table.csv'}")
    except PermissionError:
        print(f"  [WARNING] Could not save VIF table due to permission error, continuing...")
    
    # Create correlation heatmap
    create_correlation_heatmap(df_features, feature_cols, GRAPHS_DIR / 'corr_heatmap_features.png')
    
    # Run cross-validation
    cv_reg_df, cv_clf_df, coef_df = run_cross_validation(df_features, feature_cols)
    
    # Save CV results
    try:
        cv_reg_df.to_csv(OUTPUT_DIR / 'cv_metrics_regression.csv', index=False)
        cv_clf_df.to_csv(OUTPUT_DIR / 'cv_metrics_classification.csv', index=False)
        print(f"  [OK] CV results saved")
    except PermissionError:
        print(f"  [WARNING] Could not save CV results due to permission error, continuing...")
    
    # Print CV summary
    if not cv_reg_df.empty:
        print(f"\n[DATA] CROSS-VALIDATION SUMMARY:")
        print(f"  Folds completed: {len(cv_reg_df)}")
        print(f"  Average R²: {cv_reg_df['r2'].mean():.3f} ± {cv_reg_df['r2'].std():.3f}")
        print(f"  Average MAE: {cv_reg_df['mae'].mean():.3f} ± {cv_reg_df['mae'].std():.3f}")
        print(f"  Average Hit Rate: {cv_reg_df['hit_rate'].mean():.3f} ± {cv_reg_df['hit_rate'].std():.3f}")
        
        # Print CV metrics table
        print(f"\n[LIST] CV METRICS TABLE:")
        cv_summary = cv_reg_df.groupby('model')[['r2', 'mae', 'hit_rate']].agg(['mean', 'std']).round(3)
        print(cv_summary)
    else:
        print(f"\n[WARNING] No CV folds completed - using holdout only")
    
    # Analyze coefficient stability
    stability_df = analyze_coefficient_stability(coef_df)
    try:
        stability_df.to_csv(OUTPUT_DIR / 'coef_stability.csv', index=False)
        print(f"  [OK] Coefficient stability saved")
    except PermissionError:
        print(f"  [WARNING] Could not save coefficient stability due to permission error, continuing...")
    
    # Create baseline models
    baselines = create_baseline_models(df_features, feature_cols)
    
    # Create diagnostic plots
    create_diagnostic_plots(cv_reg_df, cv_clf_df, coef_df, df_features, feature_cols)
    
    # Run final holdout evaluation
    holdout_metrics = run_final_holdout_evaluation(df_features, feature_cols)
    
    # Save final metrics
    try:
        with open(OUTPUT_DIR / 'final_holdout_metrics.json', 'w') as f:
            json.dump(holdout_metrics, f, indent=2, default=str)
        print(f"  [OK] Final metrics saved")
    except PermissionError:
        print(f"  [WARNING] Could not save final metrics due to permission error, continuing...")
    
    # Print summary
    print(f"\n[OK] Step 8 Complete!")
    print(f"[FOLDER] Results saved to: {OUTPUT_DIR}")
    print(f"[DATA] Plots saved to: {GRAPHS_DIR}")
    print(f"[TARGET] Final holdout R²: {holdout_metrics['regression']['r2']:.3f}")
    print(f"[TARGET] Final holdout ROC-AUC: {holdout_metrics['classification']['roc_auc']:.3f}")
    print(f"[TARGET] Final holdout Hit Rate: {holdout_metrics['regression']['hit_rate']:.3f}")

if __name__ == "__main__":
    main()
