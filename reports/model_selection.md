# Model Selection and Tuning Process Documentation

## 1. Overview
This document outlines the model development, hyperparameter tuning, and model selection procedure for the UCI Heart Disease classification problem in accordance with Task 2 requirements.

---

## 2. Preprocessing & Feature Engineering
- **Numerical Pipeline**: `StandardScaler` applied to `['age', 'trestbps', 'chol', 'thalach', 'oldpeak']`.
- **Categorical Pipeline**: `OneHotEncoder(handle_unknown='ignore')` applied to `['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']`.
- **Encapsulation**: Wrapped inside a single `ColumnTransformer` combined with the downstream estimator inside an end-to-end Scikit-Learn `Pipeline` to prevent data leakage during cross-validation.

---

## 3. Candidate Architectures & Hyperparameter Tuning Strategy
Tuning was performed using `GridSearchCV` with **Stratified 5-Fold Cross-Validation** optimized for **ROC-AUC**:

### A. Logistic Regression (`logreg`)
- **Search Space**:
  - `classifier__C`: `[0.01, 0.1, 1.0, 10.0]`
  - `classifier__solver`: `['lbfgs', 'liblinear']`
- **Optimal Hyperparameters**: `{'classifier__C': 1.0, 'classifier__solver': 'lbfgs'}`
- **Best CV ROC-AUC**: `0.9025`

### B. Random Forest Classifier (`rf`)
- **Search Space**:
  - `classifier__n_estimators`: `[50, 100, 200]`
  - `classifier__max_depth`: `[3, 5, 8, None]`
  - `classifier__min_samples_split`: `[2, 5]`
- **Optimal Hyperparameters**: `{'classifier__max_depth': 8, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 100}`
- **Best CV ROC-AUC**: `0.9021`

---

## 4. Evaluation Metrics & Comparative Analysis

All models were evaluated across 5-fold cross-validation and on a stratified 20% holdout test set ($N=61$):

| Model | CV ROC-AUC (Mean ± Std) | CV Accuracy | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.9025 ± 0.0144** | **84.71% ± 1.00%** | **0.9665** | **88.52%** | **83.87%** | **92.86%** | **88.14%** |
| **Random Forest** | 0.9021 ± 0.0314 | 81.80% ± 3.09% | 0.9502 | 88.52% | 85.71% | 85.71% | 85.71% |

---

## 5. Model Selection Decision
- **Selected Champion**: **Logistic Regression** (`logreg`)
- **Justification**:
  1. Higher test ROC-AUC (0.9665 vs 0.9502) and higher CV ROC-AUC with lower fold-to-fold variance ($\pm 0.0144$).
  2. Superior sensitivity / recall (92.86% vs 85.71%) which is critical in clinical diagnostic support to minimize false negatives.
  3. High interpretability and fast microsecond-level inference latency suitable for containerized microservice serving.
