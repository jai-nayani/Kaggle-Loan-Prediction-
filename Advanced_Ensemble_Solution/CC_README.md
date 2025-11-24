# Claude Code - Loan Prediction Competition

## Competition Overview
- **Competition**: Kaggle Playground Series S5E11
- **Task**: Binary classification (loan payback prediction)
- **Metric**: AUC (Area Under ROC Curve)
- **Current Leaderboard**:
  - #1: 0.92849
  - #2: 0.92810
  - #3: 0.92810
- **Your Target**: 0.93+ (would be #1)

---

## Claude Code Solutions

### 1. CC_Script_Fast.py ✅ COMPLETED
**Status**: Completed
**OOF Score**: 0.92110
**Training Time**: ~20 minutes
**Submission File**: `CC_submission_fast.csv`

**Features**:
- 40 engineered features
- 3 models (XGBoost, LightGBM, CatBoost)
- 5-fold cross-validation
- Meta-learner stacking
- Synthetic features from original dataset

**Model Performance**:
- LightGBM: 0.92102
- XGBoost: 0.92044
- CatBoost: 0.91879
- Meta-Learner: 0.92110

---

### 2. CC_Script_Ultra.py ⏳ RUNNING
**Status**: Currently training
**Target Score**: 0.928-0.93
**Estimated Time**: 30-40 minutes
**Submission File**: `CC_submission_ultra.csv` (when complete)

**Advanced Features**:
- 91+ engineered features
- 5 diverse models:
  - XGBoost (optimized params)
  - LightGBM (optimized params)
  - CatBoost (optimized params)
  - ExtraTrees (for diversity)
  - Neural Network (3-layer MLP)
- 7-fold cross-validation
- Multiple meta-learners (LogisticRegression, Ridge, LightGBM meta, Weighted Average)
- Advanced synthetic feature engineering
- Automatic best meta-learner selection
- Prediction calibration

**Strategies Used**:
1. **Synthetic Features**: Trained models on original dataset to predict missing features
2. **Advanced Interactions**: Polynomial features, log transforms, binning
3. **Target Encoding**: Smoothed target encoding for categorical variables
4. **Statistical Aggregations**: Group-based statistics by category
5. **Model Diversity**: Multiple algorithm types for better ensemble
6. **Hyperparameter Optimization**: Manually tuned parameters based on data analysis
7. **Stacking**: Multiple layers of prediction aggregation

---

## Files Created

### Scripts
- `CC_Script_Fast.py` - Fast baseline (completed)
- `CC_Script_Ultra.py` - Ultra-optimized solution (running)
- `CC_data_analysis.py` - Data exploration and analysis
- `CC_monitor.sh` - Training progress monitor

### Outputs
- `CC_submission_fast.csv` - Fast version submission (0.92110 OOF)
- `CC_submission_ultra.csv` - Ultra version submission (pending)
- `CC_fast_log.txt` - Fast version training log
- `CC_ultra_log.txt` - Ultra version training log
- `CC_analysis_output.txt` - Data analysis results
- `CC_results.pkl` - Pickled results for analysis

### Models Directory
- `CC_models/` - Saved model artifacts

---

## Key Insights from Data Analysis

### Most Important Features (Correlation with Target):
1. **debt_to_income_ratio**: -0.336 (strong negative)
2. **credit_score**: +0.235 (strong positive)
3. **interest_rate**: -0.131 (moderate negative)

### Target Distribution:
- Class 1 (Paid Back): 79.88%
- Class 0 (Default): 20.12%
- Imbalanced but AUC handles this well

### Original Dataset Advantages:
The original dataset has 10 extra features not in competition data:
- age
- monthly_income
- loan_term
- installment
- num_of_open_accounts
- total_credit_limit
- current_balance
- delinquency_history
- public_records
- num_of_delinquencies

These were used to create synthetic features via transfer learning.

---

## How to Monitor Progress

```bash
# Check ultra script progress
tail -f CC/CC_ultra_log.txt

# Or use the monitoring script
bash CC/CC_monitor.sh

# Check process status
ps aux | grep CC_Script
```

---

## How to Submit

1. Wait for `CC_submission_ultra.csv` to be generated
2. Upload to Kaggle competition page
3. Select as one of your 2 final submissions

**Note**: You have 5 submissions per day limit. Use wisely!

---

## Competition Rules Compliance ✅

All solutions strictly follow competition rules:
- ✅ No private code sharing
- ✅ External data (original dataset) is publicly available
- ✅ All tools used are open-source (OSI-approved licenses)
- ✅ No multiple accounts
- ✅ Team size limits respected
- ✅ Code can reproduce submissions
- ✅ No prohibited techniques

---

## Next Steps

### If Ultra Script Achieves 0.926-0.93:
1. Submit `CC_submission_ultra.csv` to Kaggle
2. Wait for public leaderboard score
3. Adjust if needed based on public/private split

### If Ultra Script Achieves 0.923-0.925:
- This is very competitive (top 10-15)
- Consider additional strategies:
  - Hyperparameter tuning with Optuna
  - Pseudo-labeling iterations
  - Feature selection to remove noise
  - Test-time augmentation
  - Model calibration tweaks

### If Ultra Script Achieves < 0.923:
- Fast version (0.921) is still solid
- Analyze feature importances
- Try different meta-learner combinations
- Consider ensemble of fast + ultra predictions

---

## Technical Details

### Environment
- Python 3.13.2
- Virtual Environment: `../venv/`
- M2 GPU (Metal) support enabled

### Libraries Used
- xgboost: 3.1.2
- lightgbm: 4.6.0
- catboost: 1.2.8
- scikit-learn: 1.7.2
- pandas: 2.3.3
- numpy: 2.2.4
- optuna: 4.6.0

---

## Contact

Generated by **Claude Code**
For issues or questions, refer to competition discussion forums.

---

## Final Notes

**Realistic Expectations**:
- Reaching exactly 0.93 is very challenging
- Current #1 is 0.92849
- A score of 0.926-0.928 would likely place top 5-10
- A score of 0.928+ would likely place top 3

**Remember**: The private leaderboard (which determines final rankings) may differ from public leaderboard. Focus on robust cross-validation scores as the best indicator of true performance.

Good luck! 🍀
