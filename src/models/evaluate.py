"""Model evaluation, hyperparameter comparison, metrics reporting, and visual plotting."""

import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.train import DEFAULT_DATA_PATH, tune_and_train_model

DEFAULT_REPORT_PATH = Path("reports/model_evaluation.json")
DEFAULT_FIGURES_DIR = Path("reports/figures")

sns.set_theme(style="whitegrid", palette="muted")


def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
) -> dict:
    """Computes cross-validation scores and holdout test set performance across standard metrics."""
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "cv_metrics": {
            metric: {
                "mean": round(float(cv_results[f"test_{metric}"].mean()), 4),
                "std": round(float(cv_results[f"test_{metric}"].std()), 4),
            }
            for metric in scoring
        },
        "test_metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        },
    }
    return metrics


def plot_run_artifacts(
    model_type: str,
    y_test: pd.Series,
    y_pred,
    y_prob,
    figures_dir: Path,
) -> tuple[Path, Path]:
    """Generates and saves per-run ROC curve and confusion matrix PNGs. Returns their paths."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curve — {model_type}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    roc_path = figures_dir / f"{model_type}_roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_type}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    cm_path = figures_dir / f"{model_type}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print(f"Saved per-run plots: {roc_path.name}, {cm_path.name}")
    return roc_path, cm_path


def plot_model_comparison(results: dict, figures_dir: Path):
    """Generates comparison bar chart and ROC curves for candidate models."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metric Comparison Bar Chart
    metrics_list = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    
    logreg_scores = [results["logreg"]["test_metrics"][m] * 100 for m in metrics_list]
    rf_scores = [results["rf"]["test_metrics"][m] * 100 for m in metrics_list]

    x = np.arange(len(metrics_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, logreg_scores, width, label="Logistic Regression", color="#2b5c8f", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x + width/2, rf_scores, width, label="Random Forest", color="#d95f02", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Candidate Model Performance Comparison (Holdout Test Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(70, 105)
    ax.legend(loc="lower right", fontsize=10)

    for rects in [rects1, rects2]:
        for bar in rects:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart_file = figures_dir / "05_model_comparison.png"
    plt.savefig(chart_file, dpi=300)
    plt.close()
    print(f"Saved comparison chart: {chart_file}")


def run_model_comparison(
    data_path: Path = DEFAULT_DATA_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> dict:
    """Tunes candidate models (Logistic Regression vs Random Forest), evaluates, and exports report."""
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    results = {}
    for model_type in ["logreg", "rf"]:
        pipe, best_params, _, run_id = tune_and_train_model(model_type, data_path=data_path)
        eval_metrics = evaluate_pipeline(pipe, X_train, y_train, X_test, y_test)
        results[model_type] = {
            "best_hyperparameters": best_params,
            "cv_metrics": eval_metrics["cv_metrics"],
            "test_metrics": eval_metrics["test_metrics"],
        }
        test_auc = eval_metrics["test_metrics"]["roc_auc"]
        test_acc = eval_metrics["test_metrics"]["accuracy"]
        print(f"Model: {model_type} | Best Params: {best_params} | Test ROC-AUC: {test_auc:.4f} | Accuracy: {test_acc:.4f}")

        # Resume the run opened in tune_and_train_model to log eval metrics and per-run plots
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        roc_path, cm_path = plot_run_artifacts(model_type, y_test, y_pred, y_prob, figures_dir)

        cv_log = {}
        for metric, stats in eval_metrics["cv_metrics"].items():
            cv_log[f"cv_{metric}_mean"] = stats["mean"]
            cv_log[f"cv_{metric}_std"] = stats["std"]

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({f"test_{k}": v for k, v in eval_metrics["test_metrics"].items()})
            mlflow.log_metrics(cv_log)
            mlflow.log_artifact(str(roc_path))
            mlflow.log_artifact(str(cm_path))

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation report to {report_path}")

    figures_dir = Path(figures_dir)
    plot_model_comparison(results, figures_dir)

    # Log the summary comparison chart and evaluation report to the last run (rf)
    comparison_chart = figures_dir / "05_model_comparison.png"
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(comparison_chart))
        mlflow.log_artifact(str(report_path))

    return results


if __name__ == "__main__":
    run_model_comparison()
