"""
Claude Code OPTUNA-TUNED Pipeline
Hyperparameter optimization for maximum performance
Target: 0.925-0.927

Strategy:
- Optuna tuning for LightGBM and XGBoost
- 150 trials each (300 total trials)
- Best hyperparameters automatically selected
- 5-fold CV for robust evaluation
- Advanced features from Ultra script

Estimated time: 2-3 hours
Expected improvement: +0.002-0.003
"""

import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path
import pickle
import optuna

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# Configuration
BASE_PATH = Path('/Users/jaiadithyaramnayani/Desktop/Google x Kaggle')
OUTPUT_DIR = BASE_PATH / 'CC'

N_FOLDS = 5
N_TRIALS_LGB = 150  # Trials for LightGBM
N_TRIALS_XGB = 150  # Trials for XGBoost
RANDOM_STATE = 42

print("="*80)
print("CLAUDE CODE - OPTUNA HYPERPARAMETER OPTIMIZATION")
print("="*80)
print(f"LightGBM trials: {N_TRIALS_LGB}")
print(f"XGBoost trials: {N_TRIALS_XGB}")
print(f"Total trials: {N_TRIALS_LGB + N_TRIALS_XGB}")
print(f"Estimated time: 2-3 hours")
print("="*80)

# ============================================================================
# LOAD DATA & FEATURES (Using Ultra's feature engineering)
# ============================================================================

def load_and_prepare_data():
    print("\n[1/5] Loading data and creating features...")

    train = pd.read_csv(BASE_PATH / 'Loan/train.csv')
    test = pd.read_csv(BASE_PATH / 'Loan/test.csv')
    original = pd.read_csv(BASE_PATH / 'loan_dataset_20000.csv')

    # Synthetic features from original
    orig = original.copy()
    for col in ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']:
        le = LabelEncoder()
        orig[col] = le.fit_transform(orig[col].astype(str))

    combined = pd.concat([train, test], axis=0).reset_index(drop=True)

    for col in ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']:
        le = LabelEncoder()
        combined[f'{col}_enc'] = le.fit_transform(combined[col].astype(str))

    # Predict synthetic features
    synth_features = {}
    common_cols = ['annual_income', 'credit_score', 'debt_to_income_ratio', 'interest_rate', 'loan_amount']

    for feat in ['age', 'num_of_open_accounts', 'total_credit_limit', 'current_balance', 'num_of_delinquencies']:
        if feat in orig.columns:
            model = lgb.LGBMRegressor(n_estimators=50, random_state=RANDOM_STATE, verbose=-1)
            X_orig = orig[common_cols + ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']]
            model.fit(X_orig, orig[feat])

            X_comp = combined[common_cols].copy()
            for c in ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']:
                X_comp[c] = combined[f'{c}_enc']

            synth_features[f'synth_{feat}'] = model.predict(X_comp)

    # Feature engineering
    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train, test], axis=0).reset_index(drop=True)

    for feat_name, feat_vals in synth_features.items():
        df[feat_name] = feat_vals

    # Core features
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['monthly_income'] = df['annual_income'] / 12
    df['monthly_debt'] = df['monthly_income'] * df['debt_to_income_ratio']
    df['disposable_income'] = df['monthly_income'] - df['monthly_debt']
    df['total_interest'] = df['loan_amount'] * (df['interest_rate'] / 100)
    df['total_cost'] = df['loan_amount'] + df['total_interest']
    df['payment_to_income'] = df['total_cost'] / (df['annual_income'] + 1)
    df['risk_score'] = df['debt_to_income_ratio'] * df['interest_rate'] / (df['credit_score'] + 1)

    # Advanced features
    df['credit_dti'] = df['credit_score'] * (1 - df['debt_to_income_ratio'])
    df['income_credit'] = np.sqrt(df['annual_income'] * df['credit_score'])
    df['credit_squared'] = df['credit_score'] ** 2
    df['dti_squared'] = df['debt_to_income_ratio'] ** 2
    df['interest_squared'] = df['interest_rate'] ** 2
    df['log_income'] = np.log1p(df['annual_income'])
    df['log_loan'] = np.log1p(df['loan_amount'])
    df['log_credit'] = np.log1p(df['credit_score'])

    # Synthetic interactions
    if 'synth_age' in df.columns:
        df['age_income'] = df['synth_age'] / (df['annual_income'] / 10000 + 1)
        df['age_credit'] = df['synth_age'] * df['credit_score'] / 1000

    if 'synth_total_credit_limit' in df.columns:
        df['credit_util'] = df['loan_amount'] / (df['synth_total_credit_limit'] + 1)

    # Categorical encoding
    cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

    for col in cat_cols:
        le = LabelEncoder()
        df[f'{col}_le'] = le.fit_transform(df[col].astype(str))
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq)

    # Target encoding
    train_df = df[df['is_train'] == 1].copy()
    if 'loan_paid_back' in train.columns:
        train_df['target'] = train['loan_paid_back'].values

        for col in ['grade_subgrade', 'loan_purpose', 'employment_status']:
            means = train_df.groupby(col)['target'].agg(['mean', 'count'])
            smooth = 10
            smoothed = (means['count'] * means['mean'] + smooth * train_df['target'].mean()) / (means['count'] + smooth)
            df[f'{col}_target'] = df[col].map(smoothed).fillna(train_df['target'].mean())

    # Split
    train_out = df[df['is_train'] == 1].copy()
    test_out = df[df['is_train'] == 0].copy()

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

    print(f"  Features: {len(X_train.columns)}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    return X_train, y_train, X_test

# ============================================================================
# OPTUNA OBJECTIVES
# ============================================================================

def objective_lgb(trial, X, y):
    """Optuna objective for LightGBM"""

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': RANDOM_STATE,

        # Hyperparameters to tune
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    # 3-fold CV for speed during tuning
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(100, verbose=False)])

        pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, pred)
        scores.append(score)

    return np.mean(scores)

def objective_xgb(trial, X, y):
    """Optuna objective for XGBoost"""

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'verbosity': 0,
        'n_jobs': -1,
        'random_state': RANDOM_STATE,

        # Hyperparameters to tune
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    # 3-fold CV for speed during tuning
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, pred)
        scores.append(score)

    return np.mean(scores)

# ============================================================================
# TRAINING WITH BEST PARAMS
# ============================================================================

def train_with_best_params(X, y, X_test, params, model_type):
    """Train final model with best hyperparameters"""

    print(f"\n  Training final {model_type} with best params...")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(150, verbose=False)])
        else:  # xgb
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

        fold_scores.append(roc_auc_score(y_val, oof_preds[val_idx]))
        print(f"    Fold {fold+1}: {fold_scores[-1]:.5f}")

    oof_score = roc_auc_score(y, oof_preds)
    print(f"  Overall OOF: {oof_score:.5f} (CV: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")

    return oof_preds, test_preds, oof_score

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    # Load data
    X_train, y_train, X_test = load_and_prepare_data()

    # ========================================================================
    # LIGHTGBM TUNING
    # ========================================================================

    print(f"\n[2/5] Tuning LightGBM ({N_TRIALS_LGB} trials)...")
    print("  This will take ~45-60 minutes...")

    study_lgb = optuna.create_study(direction='maximize', study_name='lgb_tuning')
    study_lgb.optimize(lambda trial: objective_lgb(trial, X_train, y_train),
                       n_trials=N_TRIALS_LGB, show_progress_bar=True)

    best_params_lgb = study_lgb.best_params
    best_params_lgb.update({
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': RANDOM_STATE
    })

    print(f"\n  Best LightGBM score: {study_lgb.best_value:.5f}")
    print(f"  Best params: {best_params_lgb}")

    # Save params
    with open(OUTPUT_DIR / 'CC_optuna_lgb_params.pkl', 'wb') as f:
        pickle.dump(best_params_lgb, f)

    # ========================================================================
    # XGBOOST TUNING
    # ========================================================================

    print(f"\n[3/5] Tuning XGBoost ({N_TRIALS_XGB} trials)...")
    print("  This will take ~45-60 minutes...")

    study_xgb = optuna.create_study(direction='maximize', study_name='xgb_tuning')
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_train, y_train),
                       n_trials=N_TRIALS_XGB, show_progress_bar=True)

    best_params_xgb = study_xgb.best_params
    best_params_xgb.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'verbosity': 0,
        'n_jobs': -1,
        'random_state': RANDOM_STATE
    })

    print(f"\n  Best XGBoost score: {study_xgb.best_value:.5f}")
    print(f"  Best params: {best_params_xgb}")

    # Save params
    with open(OUTPUT_DIR / 'CC_optuna_xgb_params.pkl', 'wb') as f:
        pickle.dump(best_params_xgb, f)

    # ========================================================================
    # TRAIN FINAL MODELS WITH BEST PARAMS
    # ========================================================================

    print(f"\n[4/5] Training final models with best hyperparameters...")

    oof_lgb, test_lgb, score_lgb = train_with_best_params(
        X_train, y_train, X_test, best_params_lgb, 'lgb'
    )

    oof_xgb, test_xgb, score_xgb = train_with_best_params(
        X_train, y_train, X_test, best_params_xgb, 'xgb'
    )

    # ========================================================================
    # ENSEMBLE
    # ========================================================================

    print(f"\n[5/5] Creating ensemble...")

    # Weighted average based on OOF performance
    weight_lgb = score_lgb / (score_lgb + score_xgb)
    weight_xgb = score_xgb / (score_lgb + score_xgb)

    print(f"  LightGBM weight: {weight_lgb:.3f} (OOF: {score_lgb:.5f})")
    print(f"  XGBoost weight: {weight_xgb:.3f} (OOF: {score_xgb:.5f})")

    final_test = test_lgb * weight_lgb + test_xgb * weight_xgb
    final_oof = oof_lgb * weight_lgb + oof_xgb * weight_xgb
    final_score = roc_auc_score(y_train, final_oof)

    print(f"\n  Final ensemble OOF: {final_score:.5f}")

    # Generate submission
    submission = pd.read_csv(BASE_PATH / 'Loan/sample_submission.csv')
    submission['loan_paid_back'] = final_test
    submission.to_csv(OUTPUT_DIR / 'CC_submission_optuna.csv', index=False)

    # Save results
    results = {
        'lgb_params': best_params_lgb,
        'xgb_params': best_params_xgb,
        'lgb_score': score_lgb,
        'xgb_score': score_xgb,
        'final_score': final_score,
        'lgb_study': study_lgb,
        'xgb_study': study_xgb
    }

    with open(OUTPUT_DIR / 'CC_optuna_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\n" + "="*80)
    print("OPTUNA TUNING COMPLETE!")
    print("="*80)
    print(f"Final Score: {final_score:.5f}")
    print(f"Improvement over base: +{final_score - 0.92306:.5f}")
    print(f"Submission: {OUTPUT_DIR / 'CC_submission_optuna.csv'}")
    print("="*80)

if __name__ == "__main__":
    main()
