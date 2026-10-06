# Reproducibility Guide

This document explains how to reconstruct the exact environment, retrain all models, and view experiment results for the Heart Disease MLOps project.

---

## Requirements

- **Python 3.12**
- `pip` (comes with Python)
- No conda or Docker required for local development

---

## Environment Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd <repository-root>/Assignment

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install all pinned dependencies
pip install -r requirements.txt
```

All packages are pinned to exact versions in `requirements.txt` — the same versions used to produce the results in this repository.

---

## Preprocessing Pipeline & Reproducibility

The saved model artifacts (`models/logreg_model.joblib`, `models/rf_model.joblib`) are full scikit-learn **Pipelines** that embed both the preprocessing transformers and the classifier in a single serialised object:

```
Pipeline
├── preprocessor (ColumnTransformer)
│   ├── StandardScaler  → numerical columns (age, trestbps, chol, thalach, oldpeak)
│   └── OneHotEncoder   → categorical columns (sex, cp, fbs, restecg, exang, slope, ca, thal)
└── classifier (LogisticRegression / RandomForestClassifier)
```

Loading the artifact and calling `.predict(raw_df)` automatically applies the exact same scaling and encoding that was fitted during training — no separate scaler file, no manual feature transformation needed.

---

## Retraining Models

Run the full training + evaluation pipeline from the `Assignment/` directory:

```bash
# Trains both models (GridSearchCV), evaluates, saves joblib artifacts,
# and logs all MLflow runs (params, metrics, plots, model artifacts)
python -m src.models.evaluate
```

This produces:
- `models/logreg_model.joblib` and `models/rf_model.joblib`
- `reports/model_evaluation.json`
- `reports/figures/` — ROC curves, confusion matrices, model comparison chart
- MLflow runs under experiment `heart-disease-classification` in `mlruns/`

---

## Registering the Champion Model

After training, register the best model (auto-selected by test ROC-AUC) to the MLflow Model Registry:

```bash
python -m src.models.register_model
```

This saves the champion pipeline as `models:/HeartDiseaseClassifier/1` in the local file-backed registry. It can be loaded back anywhere with:

```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/HeartDiseaseClassifier/1")
predictions = model.predict(raw_dataframe)
```

---

## Viewing Experiment Runs

```bash
# Start the MLflow UI (from the Assignment/ directory)
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

Each run (one per model type) contains:
- **Parameters** — best GridSearch hyperparameters
- **Metrics** — `best_cv_roc_auc`, `test_accuracy`, `test_precision`, `test_recall`, `test_f1`, `test_roc_auc`
- **Artifacts** — ROC curve PNG, confusion matrix PNG, model comparison chart, and the full serialised sklearn Pipeline

---

## Running Tests

```bash
# From the Assignment/ directory
python -m pytest tests/ -v
```
