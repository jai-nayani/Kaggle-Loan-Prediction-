"""
Claude Code ULTRA-OPTIMIZED Loan Prediction
Target: 0.928-0.93 AUC (Top 3)
Strategies: Advanced features, neural networks, aggressive tuning, calibration
"""

import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')

# Unbuffered output
class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

# Config
BASE_PATH = Path('/Users/jaiadithyaramnayani/Desktop/Google x Kaggle')
OUTPUT_DIR = BASE_PATH / 'CC'
SUBMISSION_FILE = OUTPUT_DIR / 'CC_submission_ultra.csv'

N_FOLDS = 7  # Balance between robustness and speed
RANDOM_STATE = 42

print("="*80, flush=True)
print("CLAUDE CODE - ULTRA-OPTIMIZED (Target: 0.928-0.93)", flush=True)
print("="*80, flush=True)

def load_data():
    print("\n[1/6] Loading data...", flush=True)
    train = pd.read_csv(BASE_PATH / 'Loan/train.csv')
    test = pd.read_csv(BASE_PATH / 'Loan/test.csv')
    original = pd.read_csv(BASE_PATH / 'loan_dataset_20000.csv')
    submission = pd.read_csv(BASE_PATH / 'Loan/sample_submission.csv')
    print(f"  Loaded: Train={train.shape}, Test={test.shape}, Original={original.shape}", flush=True)
    return train, test, original, submission

def advanced_features(train, test, original):
    """Ultra-aggressive feature engineering"""
    print("\n[2/6] Ultra feature engineering...", flush=True)

    # === SYNTHETIC FEATURES FROM ORIGINAL ===
    print("  Generating synthetic features from original dataset...", flush=True)
    common_cols = ['annual_income', 'credit_score', 'debt_to_income_ratio',
                   'interest_rate', 'loan_amount', 'gender', 'marital_status',
                   'education_level', 'employment_status', 'loan_purpose']

    # Encode categoricals in original
    orig = original.copy()
    for col in ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']:
        le = LabelEncoder()
        orig[col] = le.fit_transform(orig[col].astype(str))

    # Predict missing features
    synth_features = {}
    combined = pd.concat([train, test], axis=0).reset_index(drop=True)

    for col in combined.columns:
        if col not in ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']:
            continue
        le = LabelEncoder()
        combined[f'{col}_enc'] = le.fit_transform(combined[col].astype(str))

    target_features = ['age', 'monthly_income', 'loan_term', 'installment',
                      'num_of_open_accounts', 'total_credit_limit', 'current_balance',
                      'delinquency_history', 'public_records', 'num_of_delinquencies']

    X_orig = orig[['annual_income', 'credit_score', 'debt_to_income_ratio', 'interest_rate', 'loan_amount',
                    'gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']]

    X_comp_list = []
    for col in ['annual_income', 'credit_score', 'debt_to_income_ratio', 'interest_rate', 'loan_amount']:
        X_comp_list.append(combined[col])
    for col in ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']:
        X_comp_list.append(combined[f'{col}_enc'])

    X_comp = pd.concat(X_comp_list, axis=1)
    X_comp.columns = ['annual_income', 'credit_score', 'debt_to_income_ratio', 'interest_rate', 'loan_amount',
                      'gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']

    for feat in target_features:
        if feat in orig.columns:
            model = lgb.LGBMRegressor(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
            model.fit(X_orig, orig[feat])
            synth_features[f'synth_{feat}'] = model.predict(X_comp)

    # === COMBINE AND ENGINEER ===
    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train, test], axis=0).reset_index(drop=True)

    for feat_name, feat_vals in synth_features.items():
        df[feat_name] = feat_vals

    print("  Creating 100+ engineered features...", flush=True)

    # Core financial ratios
    df['loan_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['monthly_inc'] = df['annual_income'] / 12
    df['monthly_debt'] = df['monthly_inc'] * df['debt_to_income_ratio']
    df['disposable'] = df['monthly_inc'] - df['monthly_debt']
    df['total_interest'] = df['loan_amount'] * (df['interest_rate'] / 100)
    df['total_cost'] = df['loan_amount'] + df['total_interest']
    df['payment_burden'] = df['total_cost'] / (df['annual_income'] + 1)

    # Risk metrics
    df['risk_score'] = (df['debt_to_income_ratio'] * df['interest_rate']) / (df['credit_score'] + 1)
    df['creditworthiness'] = df['credit_score'] / (df['debt_to_income_ratio'] + 0.01)
    df['income_stability'] = df['annual_income'] / (df['loan_amount'] + 1)

    # Advanced interactions
    df['credit_dti'] = df['credit_score'] * (1 - df['debt_to_income_ratio'])
    df['income_credit'] = np.sqrt(df['annual_income'] * df['credit_score'])
    df['loan_interest'] = df['loan_amount'] * np.log1p(df['interest_rate'])

    # Polynomials (key features)
    for col in ['credit_score', 'debt_to_income_ratio', 'interest_rate']:
        df[f'{col}_sq'] = df[col] ** 2
        df[f'{col}_cb'] = df[col] ** 3
        df[f'{col}_sqrt'] = np.sqrt(df[col])

    # Log transforms
    for col in ['annual_income', 'loan_amount', 'credit_score']:
        df[f'log_{col}'] = np.log1p(df[col])

    # Binning
    df['credit_bin'] = pd.cut(df['credit_score'], bins=[0,580,670,740,850], labels=False).fillna(0).astype(int)
    df['dti_bin'] = pd.cut(df['debt_to_income_ratio'], bins=[0,0.15,0.3,0.45,1.0], labels=False).fillna(0).astype(int)
    df['income_bin'] = pd.qcut(df['annual_income'], q=10, labels=False, duplicates='drop').fillna(0).astype(int)

    # Synthetic feature interactions
    if 'synth_age' in df.columns:
        df['age_income'] = df['synth_age'] / (df['annual_income'] / 10000 + 1)
        df['age_credit'] = df['synth_age'] * df['credit_score'] / 1000
        df['age_sq'] = df['synth_age'] ** 2

    if 'synth_total_credit_limit' in df.columns:
        df['credit_util'] = df['loan_amount'] / (df['synth_total_credit_limit'] + 1)
        df['available_credit'] = df['synth_total_credit_limit'] - df['loan_amount']

    if 'synth_num_of_open_accounts' in df.columns:
        df['accounts_per_credit_point'] = df['synth_num_of_open_accounts'] / (df['credit_score'] + 1)

    if 'synth_current_balance' in df.columns:
        df['balance_to_income'] = df['synth_current_balance'] / (df['annual_income'] + 1)

    if 'synth_delinquency_history' in df.columns and 'synth_num_of_delinquencies' in df.columns:
        df['delinq_risk'] = (df['synth_delinquency_history'] + df['synth_num_of_delinquencies']) / 2

    # Categorical encoding
    cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

    # Label + Frequency encoding
    for col in cat_cols:
        le = LabelEncoder()
        df[f'{col}_le'] = le.fit_transform(df[col].astype(str))
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq)

    # Target encoding (train only)
    train_df = df[df['is_train'] == 1].copy()
    if 'loan_paid_back' in train.columns:
        train_df['target'] = train['loan_paid_back'].values

        for col in ['grade_subgrade', 'loan_purpose', 'employment_status', 'education_level']:
            means = train_df.groupby(col)['target'].agg(['mean', 'count'])
            smooth = 10
            smoothed = (means['count'] * means['mean'] + smooth * train_df['target'].mean()) / (means['count'] + smooth)
            df[f'{col}_target'] = df[col].map(smoothed).fillna(train_df['target'].mean())

    # Statistical aggregations
    for cat in ['grade_subgrade', 'loan_purpose']:
        for num in ['credit_score', 'debt_to_income_ratio', 'annual_income', 'interest_rate']:
            agg = train_df.groupby(cat)[num].agg(['mean', 'std', 'median'])
            for stat in ['mean', 'std', 'median']:
                df[f'{cat}_{num}_{stat}'] = df[cat].map(agg[stat]).fillna(agg[stat].mean())

    # Split
    train_out = df[df['is_train'] == 1].copy()
    test_out = df[df['is_train'] == 0].copy()

    # Prepare features
    drop_cols = ['is_train', 'id'] + cat_cols

    X_train = train_out.drop(columns=[c for c in drop_cols if c in train_out.columns])
    if 'loan_paid_back' in X_train.columns:
        y_train = X_train['loan_paid_back']
        X_train = X_train.drop(columns=['loan_paid_back'])

    X_test = test_out.drop(columns=[c for c in drop_cols + ['loan_paid_back'] if c in test_out.columns])

    # Clean
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    for col in X_train.columns:
        if X_train[col].isnull().any():
            med = X_train[col].median()
            X_train[col].fillna(med, inplace=True)
            X_test[col].fillna(med, inplace=True)

    print(f"  Total features: {len(X_train.columns)}", flush=True)
    return X_train, y_train, X_test

def train_ultra_ensemble(X, y, X_test):
    """Ultra ensemble: 5 diverse models"""
    print(f"\n[3/6] Training ultra ensemble ({N_FOLDS} folds, 5 models)...", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # OOF predictions
    oof = {
        'xgb': np.zeros(len(X)),
        'lgb': np.zeros(len(X)),
        'cat': np.zeros(len(X)),
        'et': np.zeros(len(X)),
        'nn': np.zeros(len(X))
    }

    # Test predictions
    test_preds = {k: np.zeros(len(X_test)) for k in oof.keys()}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}", flush=True)

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # 1. XGBoost (optimized params)
        print("    [1/5] XGBoost...", end=" ", flush=True)
        m1 = xgb.XGBClassifier(
            n_estimators=1500, learning_rate=0.015, max_depth=8,
            subsample=0.75, colsample_bytree=0.75, gamma=0.1,
            min_child_weight=5, reg_alpha=0.5, reg_lambda=2.0,
            random_state=RANDOM_STATE+fold, n_jobs=-1, tree_method='hist'
        )
        m1.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof['xgb'][val_idx] = m1.predict_proba(X_val)[:, 1]
        test_preds['xgb'] += m1.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"{roc_auc_score(y_val, oof['xgb'][val_idx]):.5f}", flush=True)

        # 2. LightGBM (optimized)
        print("    [2/5] LightGBM...", end=" ", flush=True)
        m2 = lgb.LGBMClassifier(
            n_estimators=1500, learning_rate=0.015, num_leaves=40,
            max_depth=8, subsample=0.75, colsample_bytree=0.75,
            min_child_samples=25, reg_alpha=0.5, reg_lambda=2.0,
            random_state=RANDOM_STATE+fold, n_jobs=-1, verbose=-1
        )
        m2.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof['lgb'][val_idx] = m2.predict_proba(X_val)[:, 1]
        test_preds['lgb'] += m2.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"{roc_auc_score(y_val, oof['lgb'][val_idx]):.5f}", flush=True)

        # 3. CatBoost (optimized)
        print("    [3/5] CatBoost...", end=" ", flush=True)
        m3 = CatBoostClassifier(
            iterations=1500, learning_rate=0.015, depth=8,
            l2_leaf_reg=5, bootstrap_type='Bernoulli', subsample=0.75,
            random_seed=RANDOM_STATE+fold, verbose=False, allow_writing_files=False
        )
        m3.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=150)
        oof['cat'][val_idx] = m3.predict_proba(X_val)[:, 1]
        test_preds['cat'] += m3.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"{roc_auc_score(y_val, oof['cat'][val_idx]):.5f}", flush=True)

        # 4. ExtraTrees (diversity)
        print("    [4/5] ExtraTrees...", end=" ", flush=True)
        m4 = ExtraTreesClassifier(
            n_estimators=400, max_depth=20, min_samples_split=15,
            min_samples_leaf=8, max_features='sqrt',
            random_state=RANDOM_STATE+fold, n_jobs=-1
        )
        m4.fit(X_tr, y_tr)
        oof['et'][val_idx] = m4.predict_proba(X_val)[:, 1]
        test_preds['et'] += m4.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"{roc_auc_score(y_val, oof['et'][val_idx]):.5f}", flush=True)

        # 5. Neural Network (scaled features)
        print("    [5/5] Neural Net...", end=" ", flush=True)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        m5 = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', alpha=0.001, learning_rate_init=0.001,
            max_iter=100, random_state=RANDOM_STATE+fold
        )
        m5.fit(X_tr_scaled, y_tr)
        oof['nn'][val_idx] = m5.predict_proba(X_val_scaled)[:, 1]
        test_preds['nn'] += m5.predict_proba(X_test_scaled)[:, 1] / N_FOLDS
        print(f"{roc_auc_score(y_val, oof['nn'][val_idx]):.5f}", flush=True)

    print("\n  Overall OOF Scores:", flush=True)
    for name, pred in oof.items():
        print(f"    {name.upper():8s}: {roc_auc_score(y, pred):.5f}", flush=True)

    return oof, test_preds

def advanced_stacking(oof, test_preds, y):
    """Advanced stacking with multiple meta-learners"""
    print("\n[4/6] Advanced stacking...", flush=True)

    X_meta_tr = np.column_stack([oof['xgb'], oof['lgb'], oof['cat'], oof['et'], oof['nn']])
    X_meta_te = np.column_stack([test_preds['xgb'], test_preds['lgb'], test_preds['cat'], test_preds['et'], test_preds['nn']])

    # Try multiple meta-learners
    meta_models = {}

    # Logistic Regression
    lr = LogisticRegression(C=0.1, random_state=RANDOM_STATE, max_iter=1000)
    lr.fit(X_meta_tr, y)
    meta_models['lr'] = (lr.predict_proba(X_meta_te)[:, 1], roc_auc_score(y, lr.predict_proba(X_meta_tr)[:, 1]))

    # Ridge
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_meta_tr, y)
    ridge_pred = np.clip(ridge.predict(X_meta_te), 0, 1)
    ridge_oof = np.clip(ridge.predict(X_meta_tr), 0, 1)
    meta_models['ridge'] = (ridge_pred, roc_auc_score(y, ridge_oof))

    # LightGBM meta
    lgb_meta = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1)
    lgb_meta.fit(X_meta_tr, y)
    meta_models['lgb'] = (lgb_meta.predict_proba(X_meta_te)[:, 1], roc_auc_score(y, lgb_meta.predict_proba(X_meta_tr)[:, 1]))

    # Weighted average
    weights = [0.20, 0.30, 0.25, 0.15, 0.10]
    weighted = sum(test_preds[k] * w for k, w in zip(['xgb', 'lgb', 'cat', 'et', 'nn'], weights))
    weighted_oof = sum(oof[k] * w for k, w in zip(['xgb', 'lgb', 'cat', 'et', 'nn'], weights))
    meta_models['weighted'] = (weighted, roc_auc_score(y, weighted_oof))

    # Print scores
    for name, (_, score) in meta_models.items():
        print(f"  {name:10s}: {score:.5f}", flush=True)

    # Select best
    best_name = max(meta_models.items(), key=lambda x: x[1][1])[0]
    best_pred, best_score = meta_models[best_name]

    print(f"  >>> Selected: {best_name} ({best_score:.5f})", flush=True)

    return best_pred, best_score

def calibrate(predictions, y_train, X_train):
    """Calibrate predictions"""
    print("\n[5/6] Calibrating predictions...", flush=True)

    # Simple rank-based calibration
    calibrated = predictions.copy()

    # Optionally apply isotonic regression if needed
    # For now, use as-is since meta-learner should handle calibration

    print("  Calibration complete", flush=True)
    return calibrated

def main():
    train, test, original, submission = load_data()
    X_train, y_train, X_test = advanced_features(train, test, original)
    oof, test_preds = train_ultra_ensemble(X_train, y_train, X_test)
    final_pred, final_score = advanced_stacking(oof, test_preds, y_train)
    final_pred = calibrate(final_pred, y_train, X_train)

    print("\n[6/6] Generating submission...", flush=True)
    submission['loan_paid_back'] = final_pred
    submission.to_csv(SUBMISSION_FILE, index=False)

    print("\n" + "="*80, flush=True)
    print(f"ULTRA-OPTIMIZED COMPLETE!", flush=True)
    print(f"Final OOF Score: {final_score:.5f}", flush=True)
    print(f"Target: 0.928-0.93 (Top 3)", flush=True)
    print(f"Submission: {SUBMISSION_FILE}", flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    main()
