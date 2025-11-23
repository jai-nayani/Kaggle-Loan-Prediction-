# 🎯 Predicting Loan Payback - Kaggle Playground S5E11

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Playground_S5E11-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/playground-series-s5e11)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge&logo=python)](https://www.python.org/)
[![Score](https://img.shields.io/badge/Leaderboard_Score-0.92232-success?style=for-the-badge)](https://github.com/jai-nayani/Kaggle-Loan-Prediction-)

A comprehensive solution for the **Kaggle Playground Series Season 5, Episode 11** competition. This project demonstrates an end-to-end machine learning pipeline, from deep feature engineering to ensemble stacking, achieving a top-tier score of **0.922+**.

---

## 📊 Competition Overview
**Goal:** Predict the probability that a borrower will pay back their loan (`loan_paid_back`).  
**Metric:** ROC AUC (Area Under the Receiver Operating Characteristic Curve).  
**Data:** Synthetic dataset generated from a deep learning model trained on the original "Loan Prediction" dataset.

---

## 💡 Key Strategy & Approach

To break past the standard baseline (~0.921), we employed a multi-stage strategy focusing on **diversity** and **data integrity**.

### 1. 🛠️ Feature Engineering
We went beyond raw columns to capture hidden financial interactions:
*   **Affordability Ratios**: `loan_to_income`, `payment_to_income`, `disposable_income`.
*   **Polynomial Interactions**: Generated interaction terms between `credit_score`, `debt_to_income_ratio`, and `annual_income`.
*   **Log Transforms**: Handled skewed distributions in `annual_income` and `loan_amount`.
*   **Categorical Encoding**: Mixed **Label Encoding** (for Tree models) with **Target Encoding** (smoothed) to capture high-cardinality signal in `loan_purpose`.

### 2. 🤖 Ensemble Modeling (The "Top 3" Architecture)
Instead of relying on a single model, we stacked three powerful Gradient Boosting Machines (GBMs):

| Model | Role | Key Parameters |
| :--- | :--- | :--- |
| **LightGBM** | **The Anchor** | `learning_rate=0.005`, `n_estimators=5000`, `num_leaves=64` |
| **CatBoost** | **Categorical Expert** | Native categorical handling, `depth=8`, `l2_leaf_reg=3` |
| **XGBoost** | **Structure Expert** | `max_depth=8`, `subsample=0.7`, `colsample_bytree=0.5` |

**Blending Strategy:**  
We verified that our models were highly correlated (>0.99), so we focused on "Low & Slow" training (low learning rate, high estimators) to squeeze out maximum precision rather than blindly averaging weak models.

### 3. 🧪 Pseudo-Labeling
To bridge the gap to the top leaderboard scores (0.928), we implemented **Iterative Pseudo-Labeling**:
1.  Train initial Ensemble.
2.  Predict on Test Set.
3.  Select **Top 5%** most confident predictions (Probability > 0.99 or < 0.01).
4.  Add these "confident" rows to Training Data.
5.  **Retrain** the best model (LightGBM) on this augmented dataset.

---

## 📂 Project Structure

```
├── Loan/
│   ├── train_baseline.py       # Initial XGBoost Baseline (~0.921)
│   ├── train_advanced.py       # Advanced Stacking & Feature Engineering (~0.9216)
│   ├── train_top3.py           # Polynomials + Pseudo-Labeling Pipeline
│   ├── fix_submission.py       # 🚑 CRITICAL: Guarantees ID alignment for submission
│   ├── force_fix_submission.py # 🚑 FALLBACK: Byte-perfect recreation of valid submission
│   ├── submission_final_fix.csv # ✅ The Winning Submission File
│   └── autogluon_kaggle_script.py # Automated Stacking Script for Kaggle Notebooks
└── README.md
```

---

## 🚀 How to Reproduce

### 1. Setup Environment
```bash
pip install pandas numpy xgboost lightgbm catboost scikit-learn
```

### 2. Run the "Safe" Pipeline
We discovered that complex feature engineering can sometimes misalign row IDs. Use our "Safe" pipeline to generate a valid submission:
```bash
python Loan/fix_submission.py
```

### 3. (Optional) Automated Stacking
To push for the **0.928** score, copy the content of `Loan/autogluon_kaggle_script.py` into a Kaggle Notebook and run with GPU acceleration.

---

## 📈 Results & Learnings

*   **Baseline Score**: `0.92116`
*   **Advanced Ensemble**: `0.92166`
*   **Final "Fixed" Submission**: **`0.92232`** 🏆

**Critical Insight:**  
The biggest challenge wasn't the model accuracy, but **Submission Integrity**. Kaggle's parser is strict. We solved a "Failed to initiate scoring" error by writing a script (`force_fix_submission.py`) that acts as a "template filler," ensuring our CSV was byte-compatible with the sample submission file.

---

### 🔗 References
*   [Kaggle Competition Link](https://www.kaggle.com/competitions/playground-series-s5e11)
*   [AutoGluon Documentation](https://auto.gluon.ai/)

---
*Created by [Jai Nayani](https://github.com/jai-nayani)*

