"""CI training script — trains, evaluates, and registers the champion model.

Designed to run in a clean GitHub Actions environment. Exits non-zero on any failure.

Arguments:
    --data-path    Path to the processed CSV dataset
    --models-dir   Directory to save trained .joblib model files (default: models)
    --report-path  Path for the evaluation JSON report
                   (default: reports/model_evaluation.json)
    --figures-dir  Directory for evaluation figures (default: reports/figures)

Usage:
    python src/ml_pipeline/ci_train.py
    python src/ml_pipeline/ci_train.py --data-path data/processed/heart_disease_cleaned.csv
    python src/ml_pipeline/ci_train.py --data-path data/processed/heart_disease_cleaned.csv --models-dir models
    python src/ml_pipeline/ci_train.py --report-path reports/model_evaluation.json --figures-dir reports/figures
"""

import argparse
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # noqa: E402
from src.ml_pipeline.tasks.preprocess import preprocess_pipeline, DEFAULT_CLEAN_PATH  # noqa: E402
from src.ml_pipeline.tasks.evaluate import run_model_comparison  # noqa: E402
from src.ml_pipeline.tasks.register_model import register_champion_model  # noqa: E402

DEFAULT_DATA_PATH = DEFAULT_CLEAN_PATH
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_REPORT_PATH = Path("reports/model_evaluation.json")
DEFAULT_FIGURES_DIR = Path("reports/figures")


def main(data_path: Path, models_dir: Path, report_path: Path, figures_dir: Path) -> None:
    # Use sqlite store so CI never needs a pre-existing mlflow.db.
    # set_tracking_uri is global state; this call wins because it runs before
    # run_model_comparison() and register_champion_model() set their own URIs.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Regenerate processed CSV if absent (e.g. first CI run from clean checkout).
    if not data_path.exists():
        print(f"Processed data not found at {data_path}. Running preprocessing...")
        preprocess_pipeline()
        print("Preprocessing complete.")

    print(f"Training on: {data_path}")
    print(f"Saving models to: {models_dir}\n")

    results = run_model_comparison(
        data_path=data_path,
        report_path=report_path,
        figures_dir=figures_dir,
    )
    print(f"Evaluation report written to: {report_path}\n")

    model_uri = register_champion_model(
        report_path=report_path,
        models_dir=models_dir,
        data_path=data_path,
    )

    # Select champion by highest CV ROC-AUC and write to models/champion.txt.
    # The API reads this file at startup to resolve which .joblib to load —
    # no MLflow server needed at serve time.
    champion = max(results, key=lambda m: results[m]["cv_metrics"]["roc_auc"]["mean"])
    champion_file = models_dir / "champion.txt"
    champion_file.write_text(champion)

    print("=" * 50)
    print("CI Training Summary")
    print("=" * 50)
    # register_champion_model() registers the model with the highest test ROC-AUC.
    registered = max(results, key=lambda m: results[m]["test_metrics"]["roc_auc"])
    for model_type, info in results.items():
        marker = " ✓ champion" if model_type == champion else ""
        cv_auc = info["cv_metrics"]["roc_auc"]["mean"]
        test_auc = info["test_metrics"]["roc_auc"]
        uri = model_uri if model_type == registered else "not registered"
        print(
            f"  {model_type:8s}  CV ROC-AUC={cv_auc:.4f}  "
            f"Test ROC-AUC={test_auc:.4f}  registry={uri}{marker}"
        )
    print(f"  Champion written to: {champion_file}")
    print(f"  Registered model URI: {model_uri}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CI model training script")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the processed CSV dataset",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory to save trained .joblib model files",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to write the model evaluation JSON report",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory to save evaluation figures",
    )
    args = parser.parse_args()

    try:
        main(
            data_path=args.data_path,
            models_dir=args.models_dir,
            report_path=args.report_path,
            figures_dir=args.figures_dir,
        )
    except Exception as exc:
        print(f"ERROR: CI training failed — {exc}", file=sys.stderr)
        sys.exit(1)
