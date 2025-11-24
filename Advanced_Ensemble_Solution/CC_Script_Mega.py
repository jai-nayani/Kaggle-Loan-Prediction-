"""
Claude Code MEGA Pipeline - Single Script to 0.927+
Trains 10 diverse models, generates submissions, applies advanced blending
Target: 0.927-0.929 (Top 5)

Strategy:
- 10 diverse model configurations
- Different random seeds, features, hyperparameters
- Position-based dynamic weighting blending
- Hierarchical ensemble optimization

Estimated time: 2-3 hours
"""

import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path
import pickle

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

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

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = Path('/Users/jaiadithyaramnayani/Desktop/Google x Kaggle')
OUTPUT_DIR = BASE_PATH / 'CC'
MODELS_DIR = OUTPUT_DIR / 'CC_mega_models'
MODELS_DIR.mkdir(exist_ok=True)

N_FOLDS = 5
N_MODELS = 10  # Number of diverse models to create
RANDOM_STATE = 42

print("="*80)
print("CLAUDE CODE - MEGA PIPELINE (0.927+ Target)")
print("="*80)
print(f"Will create {N_MODELS} diverse models")
print(f"Cross-validation: {N_FOLDS} folds")
print(f"Advanced hierarchical blending enabled")
print("="*80)

# ============================================================================
# DATA LOADING & FEATURE ENGINEERING
# ============================================================================

def load_data():
    print("\n[1/4] Loading data...")
    train = pd.read_csv(BASE_PATH / 'Loan/train.csv')
    test = pd.read_csv(BASE_PATH / 'Loan/test.csv')
    original = pd.read_csv(BASE_PATH / 'loan_dataset_20000.csv')
    submission = pd.read_csv(BASE_PATH / 'Loan/sample_submission.csv')
    print(f"  Train: {train.shape}, Test: {test.shape}")
    return train, test, original, submission

def create_features(train, test, original, feature_set='full'):
    """
    Creates features with different variants for diversity
    feature_set: 'full', 'reduced', 'advanced', 'basic'
    """

    # Synthetic features from original
    common_cols = ['annual_income', 'credit_score', 'debt_to_income_ratio',
                   'interest_rate', 'loan_amount']

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
    if feature_set in ['full', 'advanced']:
        for feat in ['age', 'num_of_open_accounts', 'total_credit_limit',
                     'current_balance', 'num_of_delinquencies']:
            if feat in orig.columns:
                model = lgb.LGBMRegressor(n_estimators=50, random_state=RANDOM_STATE, verbose=-1)
                X_orig = orig[common_cols + ['gender', 'marital_status', 'education_level',
                                             'employment_status', 'loan_purpose']]
                model.fit(X_orig, orig[feat])

                X_comp = combined[common_cols].copy()
                for c in ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']:
                    X_comp[c] = combined[f'{c}_enc']

                synth_features[f'synth_{feat}'] = model.predict(X_comp)

    # Combine
    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train, test], axis=0).reset_index(drop=True)

    for feat_name, feat_vals in synth_features.items():
        df[feat_name] = feat_vals

    # Core features (all sets)
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['monthly_income'] = df['annual_income'] / 12
    df['monthly_debt'] = df['monthly_income'] * df['debt_to_income_ratio']
    df['disposable_income'] = df['monthly_income'] - df['monthly_debt']
    df['total_interest'] = df['loan_amount'] * (df['interest_rate'] / 100)
    df['total_cost'] = df['loan_amount'] + df['total_interest']
    df['payment_to_income'] = df['total_cost'] / (df['annual_income'] + 1)
    df['risk_score'] = df['debt_to_income_ratio'] * df['interest_rate'] / (df['credit_score'] + 1)

    # Advanced features
    if feature_set in ['full', 'advanced']:
        df['credit_dti'] = df['credit_score'] * (1 - df['debt_to_income_ratio'])
        df['income_credit'] = np.sqrt(df['annual_income'] * df['credit_score'])
        df['credit_squared'] = df['credit_score'] ** 2
        df['dti_squared'] = df['debt_to_income_ratio'] ** 2
        df['log_income'] = np.log1p(df['annual_income'])
        df['log_loan'] = np.log1p(df['loan_amount'])

    # Categorical encoding
    cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status',
                'loan_purpose', 'grade_subgrade']

    for col in cat_cols:
        le = LabelEncoder()
        df[f'{col}_le'] = le.fit_transform(df[col].astype(str))
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq)

    # Target encoding (train only)
    if feature_set in ['full', 'advanced', 'reduced']:
        train_df = df[df['is_train'] == 1].copy()
        if 'loan_paid_back' in train.columns:
            train_df['target'] = train['loan_paid_back'].values

            for col in ['grade_subgrade', 'loan_purpose']:
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

    return X_train, y_train, X_test

# ============================================================================
# MODEL CONFIGURATIONS (10 DIVERSE MODELS)
# ============================================================================

def get_model_configs():
    """Returns 10 diverse model configurations"""

    configs = [
        # Config 1: Aggressive LightGBM
        {
            'name': 'model_01_lgb_aggressive',
            'type': 'lgb',
            'params': {
                'n_estimators': 2000, 'learning_rate': 0.01, 'num_leaves': 50,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42
            },
            'feature_set': 'full'
        },

        # Config 2: Conservative XGBoost
        {
            'name': 'model_02_xgb_conservative',
            'type': 'xgb',
            'params': {
                'n_estimators': 1500, 'learning_rate': 0.02, 'max_depth': 6,
                'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': 123
            },
            'feature_set': 'reduced'
        },

        # Config 3: Deep CatBoost
        {
            'name': 'model_03_cat_deep',
            'type': 'cat',
            'params': {
                'iterations': 2000, 'learning_rate': 0.015, 'depth': 9,
                'l2_leaf_reg': 3, 'random_seed': 456
            },
            'feature_set': 'full'
        },

        # Config 4: Fast LightGBM
        {
            'name': 'model_04_lgb_fast',
            'type': 'lgb',
            'params': {
                'n_estimators': 1000, 'learning_rate': 0.03, 'num_leaves': 31,
                'subsample': 0.75, 'colsample_bytree': 0.75, 'random_state': 789
            },
            'feature_set': 'basic'
        },

        # Config 5: Regularized XGBoost
        {
            'name': 'model_05_xgb_regularized',
            'type': 'xgb',
            'params': {
                'n_estimators': 1800, 'learning_rate': 0.015, 'max_depth': 8,
                'reg_alpha': 1.0, 'reg_lambda': 2.0, 'random_state': 101
            },
            'feature_set': 'advanced'
        },

        # Config 6: Balanced CatBoost
        {
            'name': 'model_06_cat_balanced',
            'type': 'cat',
            'params': {
                'iterations': 1500, 'learning_rate': 0.02, 'depth': 7,
                'l2_leaf_reg': 5, 'random_seed': 202
            },
            'feature_set': 'reduced'
        },

        # Config 7: Wide LightGBM
        {
            'name': 'model_07_lgb_wide',
            'type': 'lgb',
            'params': {
                'n_estimators': 1200, 'learning_rate': 0.025, 'num_leaves': 64,
                'subsample': 0.7, 'colsample_bytree': 0.7, 'random_state': 303
            },
            'feature_set': 'full'
        },

        # Config 8: Boosted XGBoost
        {
            'name': 'model_08_xgb_boosted',
            'type': 'xgb',
            'params': {
                'n_estimators': 2500, 'learning_rate': 0.008, 'max_depth': 7,
                'subsample': 0.85, 'colsample_bytree': 0.85, 'random_state': 404
            },
            'feature_set': 'advanced'
        },

        # Config 9: ExtraTrees (diversity)
        {
            'name': 'model_09_extratrees',
            'type': 'et',
            'params': {
                'n_estimators': 400, 'max_depth': 18, 'min_samples_split': 12,
                'random_state': 505
            },
            'feature_set': 'full'
        },

        # Config 10: Ensemble LightGBM
        {
            'name': 'model_10_lgb_ensemble',
            'type': 'lgb',
            'params': {
                'n_estimators': 1600, 'learning_rate': 0.018, 'num_leaves': 40,
                'subsample': 0.82, 'colsample_bytree': 0.82, 'random_state': 606
            },
            'feature_set': 'advanced'
        }
    ]

    return configs

# ============================================================================
# TRAINING INDIVIDUAL MODELS
# ============================================================================

def train_single_model(config, X, y, X_test):
    """Train a single model configuration"""

    print(f"\n  Training {config['name']}...")

    model_type = config['type']
    params = config['params']

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params, n_jobs=-1, verbose=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(150, verbose=False)])

        elif model_type == 'xgb':
            model = xgb.XGBClassifier(**params, n_jobs=-1, tree_method='hist')
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        elif model_type == 'cat':
            model = CatBoostClassifier(**params, verbose=False, allow_writing_files=False)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=150)

        elif model_type == 'et':
            model = ExtraTreesClassifier(**params, n_jobs=-1)
            model.fit(X_tr, y_tr)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

        fold_scores.append(roc_auc_score(y_val, oof_preds[val_idx]))

    oof_score = roc_auc_score(y, oof_preds)

    print(f"    OOF Score: {oof_score:.5f} (CV: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")

    return {
        'name': config['name'],
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'oof_score': oof_score,
        'cv_scores': fold_scores
    }

# ============================================================================
# HIERARCHICAL BLENDING
# ============================================================================

def hierarchical_blend(model_results, y_train, X_test_ids):
    """
    Advanced hierarchical blending with position-based dynamic weighting
    Similar to the 0.92773 approach
    """

    print("\n[3/4] Applying hierarchical blending...")

    n_models = len(model_results)
    n_samples = len(model_results[0]['test_preds'])

    # Create prediction matrix
    test_pred_matrix = np.column_stack([r['test_preds'] for r in model_results])
    oof_pred_matrix = np.column_stack([r['oof_preds'] for r in model_results])

    # Base weights (from OOF performance)
    base_weights = np.array([r['oof_score'] for r in model_results])
    base_weights = base_weights / base_weights.sum()  # Normalize

    # Position-based dynamic weighting
    def blend_with_position_weights(pred_matrix, base_wts):
        n_samples, n_models = pred_matrix.shape
        blended = np.zeros(n_samples)

        # Sub-weights for position-based adjustment
        # Penalize extremes, reward middle positions
        if n_models <= 4:
            sub_wts = [-0.02, 0.02, 0.02, -0.02][:n_models]
        elif n_models <= 6:
            sub_wts = [-0.02, 0.01, 0.02, 0.02, 0.01, -0.02][:n_models]
        else:
            # For 10 models
            sub_wts = [-0.02, -0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.01, -0.01, -0.02][:n_models]

        for i in range(n_samples):
            sample_preds = pred_matrix[i, :]

            # Sort predictions and get ranking
            sorted_indices = np.argsort(sample_preds)[::-1]  # Descending

            # Adjust weights based on position
            adjusted_wts = np.zeros(n_models)
            for rank, model_idx in enumerate(sorted_indices):
                adjusted_wts[model_idx] = base_wts[model_idx] + sub_wts[rank]

            # Normalize
            adjusted_wts = np.maximum(adjusted_wts, 0)  # No negative weights
            adjusted_wts = adjusted_wts / adjusted_wts.sum()

            # Blend
            blended[i] = np.dot(sample_preds, adjusted_wts)

        return blended

    # Apply blending
    test_blended = blend_with_position_weights(test_pred_matrix, base_weights)
    oof_blended = blend_with_position_weights(oof_pred_matrix, base_weights)

    blend_score = roc_auc_score(y_train, oof_blended)

    # Also try simple weighted average for comparison
    simple_blend_test = test_pred_matrix @ base_weights
    simple_blend_oof = oof_pred_matrix @ base_weights
    simple_score = roc_auc_score(y_train, simple_blend_oof)

    print(f"  Position-based blend OOF: {blend_score:.5f}")
    print(f"  Simple weighted blend OOF: {simple_score:.5f}")

    if blend_score > simple_score:
        print(f"  Using position-based blend (better by {blend_score - simple_score:.5f})")
        return test_blended, blend_score
    else:
        print(f"  Using simple weighted blend (better by {simple_score - blend_score:.5f})")
        return simple_blend_test, simple_score

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    # Load data
    train, test, original, submission = load_data()

    # Get model configurations
    configs = get_model_configs()

    print(f"\n[2/4] Training {len(configs)} diverse models...")

    # Train all models
    all_results = []

    for i, config in enumerate(configs):
        print(f"\nModel {i+1}/{len(configs)}: {config['name']}")

        # Create features for this configuration
        X_train, y_train, X_test = create_features(
            train, test, original,
            feature_set=config['feature_set']
        )

        print(f"  Feature set: {config['feature_set']} ({len(X_train.columns)} features)")

        # Train model
        result = train_single_model(config, X_train, y_train, X_test)
        all_results.append(result)

        # Save individual submission
        sub_indiv = submission.copy()
        sub_indiv['loan_paid_back'] = result['test_preds']
        sub_indiv.to_csv(MODELS_DIR / f"{config['name']}.csv", index=False)
        print(f"  Saved: {MODELS_DIR / config['name']}.csv")

    # Print summary
    print("\n" + "="*80)
    print("INDIVIDUAL MODEL SCORES:")
    print("="*80)
    for r in sorted(all_results, key=lambda x: x['oof_score'], reverse=True):
        print(f"  {r['name']:30s}: {r['oof_score']:.5f}")

    # Hierarchical blending
    X_train_final, y_train, X_test_final = create_features(train, test, original, 'full')
    final_preds, final_score = hierarchical_blend(all_results, y_train, test['id'])

    # Generate final submission
    print("\n[4/4] Generating final submission...")
    submission['loan_paid_back'] = final_preds
    final_file = OUTPUT_DIR / 'CC_submission_mega.csv'
    submission.to_csv(final_file, index=False)

    print("\n" + "="*80)
    print("MEGA PIPELINE COMPLETE!")
    print("="*80)
    print(f"Final Blended OOF Score: {final_score:.5f}")
    print(f"Number of models: {len(all_results)}")
    print(f"Final submission: {final_file}")
    print(f"Individual models: {MODELS_DIR}/")
    print("="*80)

    # Save results
    results_summary = {
        'individual_results': all_results,
        'final_score': final_score,
        'final_predictions': final_preds
    }

    with open(OUTPUT_DIR / 'CC_mega_results.pkl', 'wb') as f:
        pickle.dump(results_summary, f)

    print("\nResults saved. Ready to submit!")

if __name__ == "__main__":
    main()
