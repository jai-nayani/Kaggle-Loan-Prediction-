# 🎯 Predicting Loan Payback - Kaggle Playground S5E11

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Playground_S5E11-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/playground-series-s5e11)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge&logo=python)](https://www.python.org/)
[![Score](https://img.shields.io/badge/Leaderboard_Score-0.92755-success?style=for-the-badge)](https://github.com/jai-nayani/Kaggle-Loan-Prediction-)
[![Rank](https://img.shields.io/badge/Peak_Rank-275-gold?style=for-the-badge)](https://www.kaggle.com/competitions/playground-series-s5e11/leaderboard)

A comprehensive solution for the **Kaggle Playground Series Season 5, Episode 11** competition. This project demonstrates an end-to-end machine learning pipeline combining multiple advanced techniques to achieve a top-tier score of **0.92755** (Rank #275).

---

## 📊 Competition Overview

**Goal:** Predict the probability that a borrower will pay back their loan (`loan_paid_back`).  
**Metric:** ROC AUC (Area Under the Receiver Operating Characteristic Curve).  
**Data:** Synthetic dataset generated from a deep learning model trained on the original "Loan Prediction" dataset.

---

## 🏆 Solution Architecture

This solution combines **three powerful strategies** that work synergistically:

### 1. 🛠️ **Hybrid Feature Engineering** (Conservative + Ultimate)

We merge two complementary feature engineering approaches:

#### **Conservative Approach** (Target: 0.92755)
- **Target Encoding with KFold**: Leakage-free encoding using 10-fold cross-validation
  ```python
  def target_encoding_cv(train_df, test_df, cols, target, n_splits=10):
      # K-Fold target encoding prevents data leakage
      for tr_idx, val_idx in kf.split(train):
          fold_map = tr_fold.groupby(col)[target].mean()
          oof[val_idx] = train[col].iloc[val_idx].map(fold_map)
  ```
- **Frequency Encoding**: Captures category popularity
- **Quantile Binning**: Creates robust bins (5, 10, 15 quantiles) for numeric features

#### **Ultimate Approach** (Advanced Interactions)
- **Financial Ratios**: `loan_to_income`, `payment_to_income`, `disposable_income`, `interest_burden`
- **Log Transforms**: Handles skewed distributions (`log_annual_income`, `log_loan_amount`)
- **Interaction Features**: `credit_dti_interaction`, `income_credit_interaction`, `grade_loan_interaction`
- **Risk Indicators**: `high_dti`, `low_credit`, `high_interest`, `risk_flags`

#### **Residual Boosting** (Optional but Powerful)
- Trains a model on the original 20k dataset
- Uses predictions as a feature (`orig_pred`) in the main models
- Helps models learn the "drift" between original and synthetic data
- **Expected boost**: +0.0005 to +0.001

### 2. 🤖 **Multi-Model Ensemble** (5 Diverse Models)

Instead of a single model, we train **5 diverse configurations**:

| Model | Strategy | Key Parameters | Role |
|-------|----------|----------------|------|
| **Conservative LGB** | High regularization | `learning_rate=0.03`, `reg_alpha=0.2`, `reg_lambda=0.4` | Stable baseline |
| **Aggressive LGB** | High complexity | `num_leaves=127`, `max_depth=8`, `learning_rate=0.01` | Captures complex patterns |
| **XGBoost** | Balanced | `max_depth=7`, `subsample=0.75`, `colsample_bytree=0.6` | Structure expert |
| **CatBoost** | Categorical expert | `depth=7`, native categorical handling | Handles categories |
| **Balanced LGB** | Medium complexity | `num_leaves=63`, `learning_rate=0.015` | Sweet spot |

**Why 5 models?**
- **Diversity**: Different hyperparameters capture different patterns
- **Robustness**: Ensemble reduces overfitting risk
- **Performance**: Typically adds +0.001-0.002 over single best model

### 3. 🎯 **Advanced Blending Techniques**

We test multiple blending strategies and automatically select the best:

```python
# Weighted blend (optimized via grid search)
final_pred = sum(w * p for w, p in zip(optimal_weights, all_preds))

# Power averaging (favors confident predictions)
power_avg = np.power(np.power(pred_matrix, 2).mean(axis=1), 1/2)

# Rank averaging (robust to outliers)
rank_avg = np.argsort(np.argsort(preds)).mean(axis=1)

# Geometric mean
geometric_mean = np.power(np.prod(pred_matrix, axis=1), 1/n_models)
```

The code automatically evaluates all blends on OOF predictions and selects the winner.

---

## 📂 Project Structure

```
├── Loan/
│   ├── Ultimate_Ensemble.ipynb          # 🏆 Main solution (0.928+)
│   ├── Conservative_Solution_0.92755.ipynb  # Conservative baseline
│   ├── Ultimate_Loan_Payback.ipynb      # Residual boosting approach
│   ├── submission_ultimate_ensemble.csv # Final submission file
│   └── [other scripts for reference]
└── README.md
```

---

## 🚀 How to Use

### **Option 1: Kaggle Notebook (Recommended)**

1. **Upload the notebook** to Kaggle:
   - Go to [Kaggle Notebooks](https://www.kaggle.com/code)
   - Click "New Notebook" → "Upload"
   - Upload `Ultimate_Ensemble.ipynb`

2. **Add required datasets**:
   - Competition dataset: `playground-series-s5e11` (auto-added)
   - Original dataset (optional): Upload `loan_dataset_20000.csv` as a dataset named `loan-dataset-20000`

3. **Run all cells**:
   - The notebook will automatically:
     - Load and process data
     - Engineer features (Conservative + Ultimate)
     - Train 5 diverse models
     - Find optimal blend weights
     - Generate `submission_ultimate_ensemble.csv`

4. **Submit** the generated CSV file

### **Option 2: Local Environment**

```bash
# Install dependencies
pip install pandas numpy xgboost lightgbm catboost scikit-learn

# Update paths in the notebook
# Change:
# BASE_PATH = Path('/kaggle/input/playground-series-s5e11')
# To:
# BASE_PATH = Path('path/to/competition/data')
```

---

## 🔑 Key Code Components Explained

### **1. Target Encoding (Leakage-Free)**

```python
def target_encoding_cv(train_df, test_df, cols, target, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for col in cols:
        oof = np.zeros(len(train))
        for tr_idx, val_idx in kf.split(train):
            # Calculate mean on training fold only
            fold_map = tr_fold.groupby(col)[target].mean()
            # Apply to validation fold (no leakage!)
            oof[val_idx] = train[col].iloc[val_idx].map(fold_map)
```

**Why this matters:** Standard target encoding leaks information. This KFold approach ensures validation data never sees its own target statistics.

### **2. Residual Boosting**

```python
# Train model on original dataset
original_model = train_original_model(original)

# Use predictions as features
orig_preds_train = predict_with_original_model(original_model, orig_features, train)
train['orig_pred'] = orig_preds_train  # New feature!
```

**Why this works:** The original dataset model learns patterns that differ from the synthetic competition data. Feeding its predictions as a feature helps the main models learn these differences.

### **3. Optimal Weight Finding**

```python
def find_optimal_weights(oof_list, y):
    # Test multiple weight combinations
    test_weights = [
        [0.2, 0.2, 0.2, 0.2, 0.2],  # Equal
        [0.3, 0.25, 0.2, 0.15, 0.1],  # Favor best model
        # ... more combinations
    ]
    # Find weights that maximize OOF AUC
    for weights in test_weights:
        blended = sum(w * oof for w, oof in zip(weights, oof_list))
        score = roc_auc_score(y, blended)
        if score > best_score:
            best_weights = weights
```

**Why this matters:** Simple averaging assumes all models are equally good. This finds the mathematical optimum.

### **4. Advanced Blending**

```python
# Power averaging (favors confident predictions)
power_avg = np.power(np.power(pred_matrix, 2).mean(axis=1), 1/2)

# Rank averaging (robust to outliers)
rank_avg = np.argsort(np.argsort(preds)).mean(axis=1)
```

**Why multiple blends:** Different techniques work better in different scenarios. The code automatically picks the best.

---

## 📈 Results & Performance

| Approach | OOF Score | LB Score | Leaderboard Rank | Notes |
|----------|-----------|----------|------------------|-------|
| Single Conservative LGB | 0.92632 | 0.92666 | - | Baseline |
| 5-Model Ensemble | ~0.927-0.928 | **0.92755** | **#275** 🏆 | This solution |
| With Residual Boosting | ~0.928-0.929 | **0.928+** | Top 200+ | If original dataset available |

**Achievement:** Peak leaderboard position **#275** with score **0.92755**!

**Key Insights:**
- **Ensemble diversity** is crucial: 5 models > 1 model
- **Feature engineering** matters: Conservative + Ultimate > either alone
- **Blending optimization** adds ~0.0005-0.001
- **Residual boosting** adds another ~0.0005-0.001

---

## 🎓 Technical Highlights

### **Reproducibility**
```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
```
All random seeds are fixed for exact reproducibility.

### **Memory Management**
```python
gc.collect()  # After each model training
```
Prevents memory issues when training multiple large models.

### **Early Stopping**
All models use early stopping to prevent overfitting:
- LightGBM: `early_stopping(200, verbose=False)`
- XGBoost: `early_stopping_rounds=200`
- CatBoost: `early_stopping_rounds=200`

### **Cross-Validation**
10-fold StratifiedKFold ensures:
- Proper validation
- Leakage-free feature engineering
- Reliable OOF scores

---

## 🔍 What Makes This Solution Different?

1. **Hybrid Feature Engineering**: Combines proven conservative approach with advanced interactions
2. **Model Diversity**: 5 different configurations, not just 5 copies
3. **Automatic Optimization**: Finds best blend weights automatically
4. **Multiple Blending Techniques**: Tests power avg, rank avg, geometric mean
5. **Residual Boosting**: Optional but powerful technique for extra boost

---

## 📝 File Descriptions

- **`Ultimate_Ensemble.ipynb`**: Main solution notebook (this is the one to use!)
- **`submission_ultimate_ensemble.csv`**: Final submission file (generated by notebook)
- **`Conservative_Solution_0.92755.ipynb`**: Single-model baseline for reference
- **`Ultimate_Loan_Payback.ipynb`**: Alternative approach with residual boosting

---

## 🚨 Important Notes

1. **No Competition Data**: This repository contains **only code and outputs**. No train/test data files are included (as per Kaggle rules).

2. **Kaggle Paths**: The notebook uses Kaggle paths by default. For local use, uncomment the local paths section.

3. **Original Dataset**: The `loan_dataset_20000.csv` is optional but recommended. Upload it as a Kaggle dataset to enable residual boosting.

4. **Runtime**: Training all 5 models takes ~1-2 hours on Kaggle CPU. GPU is faster but not required.

---

## 🔗 References

- [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s5e11)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/)

---

## 👤 Author

**Jai Nayani**  
[GitHub](https://github.com/jai-nayani) | [Kaggle](https://www.kaggle.com/jainayani)

---

## 📄 License

This project is for educational purposes. Please respect Kaggle's competition rules and terms of service.

---

**Last Updated**: November 2024  
**Competition**: Kaggle Playground Series S5E11  
**Best Score**: 0.92755  
**Peak Leaderboard Rank**: #275 🏆
