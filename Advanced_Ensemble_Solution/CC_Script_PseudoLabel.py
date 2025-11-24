"""
Claude Code - PSEUDO-LABELING PIPELINE
Use high-confidence test predictions to augment training
Target: +0.001-0.002 improvement

Strategy:
- Train initial model
- Get confident predictions on test set
- Add as pseudo-labels to training
- Retrain with augmented data
- Iterate

Time: ~1 hour
"""

import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

BASE_PATH = Path('/Users/jaiadithyaramnayani/Desktop/Google x Kaggle')
OUTPUT_DIR = BASE_PATH / 'CC'

N_FOLDS = 5
CONFIDENCE_THRESHOLD = 0.99  # Very high confidence
N_ITERATIONS = 3
RANDOM_STATE = 42

print("="*80)
print("CLAUDE CODE - PSEUDO-LABELING PIPELINE")
print("="*80)
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"Iterations: {N_ITERATIONS}")
print("="*80)

def pseudo_label_pipeline(X_train, y_train, X_test):
    """Pseudo-labeling iterations"""

    print("\n[1/2] Initial model training...")

    # Initial model
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    test_preds_init = np.zeros(len(X_test))
    oof_preds_init = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.01, num_leaves=40,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
            verbose=-1, n_jobs=-1
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(150, verbose=False)])

        oof_preds_init[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds_init += model.predict_proba(X_test)[:, 1] / N_FOLDS

    initial_score = roc_auc_score(y_train, oof_preds_init)
    print(f"  Initial OOF Score: {initial_score:.5f}")

    # Pseudo-labeling iterations
    print(f"\n[2/2] Pseudo-labeling ({N_ITERATIONS} iterations)...")

    best_score = initial_score
    best_preds = test_preds_init.copy()

    X_aug = X_train.copy()
    y_aug = y_train.copy()

    for iteration in range(N_ITERATIONS):
        print(f"\n  Iteration {iteration+1}/{N_ITERATIONS}")

        # Find high-confidence predictions
        high_conf_mask = (test_preds_init >= CONFIDENCE_THRESHOLD) | (test_preds_init <= (1 - CONFIDENCE_THRESHOLD))
        n_pseudo = high_conf_mask.sum()

        print(f"    High-confidence samples: {n_pseudo}/{len(X_test)}")

        if n_pseudo == 0:
            print("    No high-confidence predictions. Stopping.")
            break

        # Create pseudo-labels
        X_pseudo = X_test[high_conf_mask].copy()
        y_pseudo = pd.Series((test_preds_init[high_conf_mask] >= 0.5).astype(int))

        # Augment training data
        X_aug = pd.concat([X_train, X_pseudo], axis=0).reset_index(drop=True)
        y_aug = pd.concat([y_train, y_pseudo], axis=0).reset_index(drop=True)

        print(f"    Augmented training size: {len(X_aug)} (original: {len(X_train)})")

        # Retrain
        test_preds_new = np.zeros(len(X_test))
        oof_preds_new = np.zeros(len(X_train))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Add pseudo-labels to training fold
            X_tr_aug = pd.concat([X_tr, X_pseudo], axis=0).reset_index(drop=True)
            y_tr_aug = pd.concat([y_tr, y_pseudo], axis=0).reset_index(drop=True)

            model = lgb.LGBMClassifier(
                n_estimators=2000, learning_rate=0.01, num_leaves=40,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE + iteration,
                verbose=-1, n_jobs=-1
            )
            model.fit(X_tr_aug, y_tr_aug, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(150, verbose=False)])

            oof_preds_new[val_idx] = model.predict_proba(X_val)[:, 1]
            test_preds_new += model.predict_proba(X_test)[:, 1] / N_FOLDS

        new_score = roc_auc_score(y_train, oof_preds_new)
        improvement = new_score - best_score

        print(f"    New OOF Score: {new_score:.5f} (improvement: {improvement:+.5f})")

        if new_score > best_score:
            best_score = new_score
            best_preds = test_preds_new.copy()
            test_preds_init = test_preds_new.copy()  # Update for next iteration
            print(f"    ✓ Improvement! New best: {best_score:.5f}")
        else:
            print(f"    ✗ No improvement. Keeping previous best.")
            break

    print(f"\n  Final Score: {best_score:.5f}")
    print(f"  Total improvement: {best_score - initial_score:+.5f}")

    return best_preds, best_score

def main():
    print("\nLoading pre-computed features...")

    # Load from advanced features script output
    # If not available, load from ultra
    try:
        # Try to load from a previous run
        import pickle
        with open(OUTPUT_DIR / 'CC_advanced_features.pkl', 'rb') as f:
            data = pickle.load(f)
            X_train, y_train, X_test = data['X_train'], data['y_train'], data['X_test']
        print("  Loaded advanced features")
    except:
        # Fallback: create basic features
        print("  Creating features...")
        from CC_Script_AdvancedFeatures import create_extreme_features

        train = pd.read_csv(BASE_PATH / 'Loan/train.csv')
        test = pd.read_csv(BASE_PATH / 'Loan/test.csv')
        original = pd.read_csv(BASE_PATH / 'loan_dataset_20000.csv')

        X_train, y_train, X_test = create_extreme_features(train, test, original)

    print(f"  Features: {len(X_train.columns)}")

    final_pred, final_score = pseudo_label_pipeline(X_train, y_train, X_test)

    print("\n[3/3] Generating submission...")
    submission = pd.read_csv(BASE_PATH / 'Loan/sample_submission.csv')
    submission['loan_paid_back'] = final_pred
    submission.to_csv(OUTPUT_DIR / 'CC_submission_pseudo.csv', index=False)

    print("\n" + "="*80)
    print("PSEUDO-LABELING COMPLETE!")
    print("="*80)
    print(f"Final Score: {final_score:.5f}")
    print(f"Submission: {OUTPUT_DIR / 'CC_submission_pseudo.csv'}")
    print("="*80)

if __name__ == "__main__":
    main()
