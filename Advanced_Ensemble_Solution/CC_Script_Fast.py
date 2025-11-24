"""
Claude Code FAST Loan Prediction Script
Target: 0.93+ AUC Score (Optimized for Speed)
Author: Claude Code
"""

import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# Force unbuffered output
class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

# Configuration
BASE_PATH = Path('/Users/jaiadithyaramnayani/Desktop/Google x Kaggle')
OUTPUT_DIR = BASE_PATH / 'CC'
SUBMISSION_FILE = OUTPUT_DIR / 'CC_submission_fast.csv'

N_FOLDS = 5  # Reduced for speed
N_ESTIMATORS = 1000  # Reduced for speed
RANDOM_STATE = 42

print("="*80, flush=True)
print("CLAUDE CODE - FAST LOAN PREDICTION", flush=True)
print("="*80, flush=True)
print(f"Folds: {N_FOLDS} | Estimators: {N_ESTIMATORS}", flush=True)
print("="*80, flush=True)

def load_and_prep_data():
    """Load and prepare all data"""
    print("\n[1/5] Loading data...", flush=True)

    train = pd.read_csv(BASE_PATH / 'Loan/train.csv')
    test = pd.read_csv(BASE_PATH / 'Loan/test.csv')
    original = pd.read_csv(BASE_PATH / 'loan_dataset_20000.csv')
    submission = pd.read_csv(BASE_PATH / 'Loan/sample_submission.csv')

    print(f"  Train: {train.shape} | Test: {test.shape}", flush=True)

    return train, test, original, submission

def create_features(train, test, original):
    """Comprehensive feature engineering"""
    print("\n[2/5] Feature engineering...", flush=True)

    # Synthetic features from original dataset
    print("  Creating synthetic features...", flush=True)
    common_features = ['annual_income', 'credit_score', 'debt_to_income_ratio',
                      'interest_rate', 'loan_amount']

    # Train simple models on original to predict extra features
    extra_features = {}
    for feat in ['age', 'num_of_open_accounts', 'total_credit_limit',
                 'current_balance', 'num_of_delinquencies']:
        model = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.1,
                                 random_state=RANDOM_STATE, verbose=-1)
        model.fit(original[common_features], original[feat])

        combined = pd.concat([train, test], axis=0)
        extra_features[f'synth_{feat}'] = model.predict(combined[common_features])

    # Combine datasets
    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train, test], axis=0).reset_index(drop=True)

    # Add synthetic features
    for feat_name, feat_values in extra_features.items():
        df[feat_name] = feat_values

    print("  Engineering features...", flush=True)

    # Core ratios
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['monthly_income'] = df['annual_income'] / 12
    df['monthly_debt'] = df['monthly_income'] * df['debt_to_income_ratio']
    df['disposable_income'] = df['monthly_income'] - df['monthly_debt']
    df['interest_amount'] = df['loan_amount'] * (df['interest_rate'] / 100)
    df['total_cost'] = df['loan_amount'] + df['interest_amount']
    df['payment_to_income'] = df['total_cost'] / (df['annual_income'] + 1)
    df['risk_score'] = df['debt_to_income_ratio'] * df['interest_rate'] / (df['credit_score'] + 1)

    # Interactions
    df['credit_dti_interact'] = df['credit_score'] * (1 - df['debt_to_income_ratio'])
    df['income_credit_interact'] = df['annual_income'] * df['credit_score']

    # Polynomials of key features
    df['credit_squared'] = df['credit_score'] ** 2
    df['dti_squared'] = df['debt_to_income_ratio'] ** 2

    # Log transforms
    df['log_income'] = np.log1p(df['annual_income'])
    df['log_loan'] = np.log1p(df['loan_amount'])

    # Binning
    df['credit_bin'] = pd.cut(df['credit_score'], bins=[0,600,700,800,850], labels=False).fillna(0).astype(int)
    df['dti_bin'] = pd.cut(df['debt_to_income_ratio'], bins=[0,0.2,0.4,0.6,1.0], labels=False).fillna(0).astype(int)

    # Categorical encoding
    cat_cols = ['gender', 'marital_status', 'education_level',
                'employment_status', 'loan_purpose', 'grade_subgrade']

    # Label encoding
    for col in cat_cols:
        le = LabelEncoder()
        df[f'{col}_le'] = le.fit_transform(df[col].astype(str))

    # Frequency encoding
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq)

    # Target encoding (only on train)
    train_df = df[df['is_train'] == 1].copy()
    if 'loan_paid_back' in train.columns:
        train_df['target'] = train['loan_paid_back'].values

        for col in ['grade_subgrade', 'loan_purpose']:
            target_mean = train_df.groupby(col)['target'].agg(['mean', 'count'])
            smoothing = 10
            smooth = (target_mean['count'] * target_mean['mean'] + smoothing * train_df['target'].mean()) / (target_mean['count'] + smoothing)
            df[f'{col}_target'] = df[col].map(smooth).fillna(train_df['target'].mean())

    # Split back
    train_out = df[df['is_train'] == 1].copy()
    test_out = df[df['is_train'] == 0].copy()

    # Prepare features
    drop_cols = ['is_train', 'id'] + cat_cols

    X_train = train_out.drop(columns=[c for c in drop_cols if c in train_out.columns])
    if 'loan_paid_back' in X_train.columns:
        y_train = X_train['loan_paid_back']
        X_train = X_train.drop(columns=['loan_paid_back'])

    X_test = test_out.drop(columns=[c for c in drop_cols + ['loan_paid_back'] if c in test_out.columns])

    # Handle NaN/inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    for col in X_train.columns:
        if X_train[col].isnull().any():
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)

    print(f"  Features: {len(X_train.columns)}", flush=True)

    return X_train, y_train, X_test

def train_models(X, y, X_test):
    """Train ensemble models"""
    print(f"\n[3/5] Training models ({N_FOLDS}-fold CV)...", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_xgb = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))

    test_xgb = np.zeros(len(X_test))
    test_lgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}", flush=True)

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # XGBoost
        print("    XGBoost...", end=" ", flush=True)
        xgb_model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS, learning_rate=0.02, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, tree_method='hist'
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        test_xgb += xgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"AUC: {roc_auc_score(y_val, oof_xgb[val_idx]):.5f}", flush=True)

        # LightGBM
        print("    LightGBM...", end=" ", flush=True)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS, learning_rate=0.02, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        test_lgb += lgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"AUC: {roc_auc_score(y_val, oof_lgb[val_idx]):.5f}", flush=True)

        # CatBoost
        print("    CatBoost...", end=" ", flush=True)
        cat_model = CatBoostClassifier(
            iterations=N_ESTIMATORS, learning_rate=0.02, depth=7,
            l2_leaf_reg=3, random_seed=RANDOM_STATE, verbose=False,
            allow_writing_files=False
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
        oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        test_cat += cat_model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"AUC: {roc_auc_score(y_val, oof_cat[val_idx]):.5f}", flush=True)

    # Overall scores
    print(f"\n  Overall OOF Scores:", flush=True)
    print(f"    XGBoost:  {roc_auc_score(y, oof_xgb):.5f}", flush=True)
    print(f"    LightGBM: {roc_auc_score(y, oof_lgb):.5f}", flush=True)
    print(f"    CatBoost: {roc_auc_score(y, oof_cat):.5f}", flush=True)

    return {
        'oof': {'xgb': oof_xgb, 'lgb': oof_lgb, 'cat': oof_cat},
        'test': {'xgb': test_xgb, 'lgb': test_lgb, 'cat': test_cat}
    }

def blend_predictions(oof_preds, test_preds, y):
    """Blend predictions using stacking"""
    print("\n[4/5] Blending predictions...", flush=True)

    # Create meta-features
    X_meta = np.column_stack([oof_preds['xgb'], oof_preds['lgb'], oof_preds['cat']])
    X_meta_test = np.column_stack([test_preds['xgb'], test_preds['lgb'], test_preds['cat']])

    # Train meta-learner
    meta = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    meta.fit(X_meta, y)

    meta_pred = meta.predict_proba(X_meta_test)[:, 1]
    meta_oof = meta.predict_proba(X_meta)[:, 1]
    meta_score = roc_auc_score(y, meta_oof)

    # Also try weighted average
    weights = [0.25, 0.40, 0.35]
    weighted_pred = (test_preds['xgb'] * weights[0] +
                     test_preds['lgb'] * weights[1] +
                     test_preds['cat'] * weights[2])
    weighted_oof = (oof_preds['xgb'] * weights[0] +
                    oof_preds['lgb'] * weights[1] +
                    oof_preds['cat'] * weights[2])
    weighted_score = roc_auc_score(y, weighted_oof)

    print(f"  Meta-learner: {meta_score:.5f}", flush=True)
    print(f"  Weighted avg: {weighted_score:.5f}", flush=True)

    if meta_score > weighted_score:
        print(f"  Using meta-learner", flush=True)
        return meta_pred, meta_score
    else:
        print(f"  Using weighted average", flush=True)
        return weighted_pred, weighted_score

def main():
    train, test, original, submission = load_and_prep_data()
    X_train, y_train, X_test = create_features(train, test, original)
    predictions = train_models(X_train, y_train, X_test)
    final_pred, final_score = blend_predictions(predictions['oof'], predictions['test'], y_train)

    print("\n[5/5] Generating submission...", flush=True)
    submission['loan_paid_back'] = final_pred
    submission.to_csv(SUBMISSION_FILE, index=False)

    print("\n" + "="*80, flush=True)
    print(f"COMPLETE! Final OOF Score: {final_score:.5f}", flush=True)
    print(f"Submission: {SUBMISSION_FILE}", flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    main()
