import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
TRAIN_PATH = 'Loan/train.csv'
TEST_PATH = 'Loan/test.csv'
SUBMISSION_PATH = 'Loan/sample_submission.csv'
OUTPUT_PATH = 'Loan/submission_safe.csv'

def run_safe_pipeline():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    
    # Store IDs explicitly
    test_ids = test['id'].values
    
    y = train['loan_paid_back']
    train = train.drop(columns=['loan_paid_back'])
    
    print("Preprocessing...")
    train['is_train'] = 1
    test['is_train'] = 0
    
    # Concatenate
    full_df = pd.concat([train, test], axis=0, ignore_index=True)
    
    # --- Feature Engineering (Simplified to ensure stability) ---
    full_df['loan_to_income'] = full_df['loan_amount'] / (full_df['annual_income'] + 1)
    
    # Simple Encoding
    cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
    for col in cat_cols:
        le = LabelEncoder()
        full_df[col] = full_df[col].astype(str)
        full_df[f'{col}_enc'] = le.fit_transform(full_df[col])
        
    drop_cols = cat_cols + ['id', 'is_train']
    
    # Split
    X = full_df[full_df['is_train'] == 1].drop(columns=drop_cols)
    X_test = full_df[full_df['is_train'] == 0].drop(columns=drop_cols)
    
    print(f"Training Simple LGBM on {X.shape}...")
    
    # Train single robust model just to pass submission check
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        objective='binary',
        metric='auc',
        random_state=42,
        verbose=-1
    )
    
    model.fit(X, y)
    preds = model.predict_proba(X_test)[:, 1]
    
    print("Creating Submission...")
    
    # Create DataFrame directly from IDs and Preds
    sub_df = pd.DataFrame({
        'id': test_ids,
        'loan_paid_back': preds
    })
    
    # Verify Length
    expected_len = 254569
    if len(sub_df) != expected_len:
        print(f"FATAL ERROR: Output length {len(sub_df)} does not match expected {expected_len}")
        return
        
    # Save
    sub_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Safe submission saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_safe_pipeline()

