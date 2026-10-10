"""CI training script — trains both models and logs to a file-based MLflow store.

Designed to run in a clean GitHub Actions environment. Sets the MLflow tracking
URI to file:./mlruns (overriding the sqlite:///mlflow.db default in train.py)
so no pre-existing database is needed. Exits non-zero on any failure.

Usage:
    python src/ci_train.py
    python src/ci_train.py --data-path data/processed/heart_disease_cleaned.csv
    python src/ci_train.py --data-path data/processed/heart_disease_cleaned.csv --models-dir models
"""

import argparse
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
from src.data.preprocess import preprocess_pipeline, DEFAULT_CLEAN_PATH  # noqa: E402
from src.models.train import tune_and_train_model  # noqa: E402

DEFAULT_DATA_PATH = DEFAULT_CLEAN_PATH
DEFAULT_MODELS_DIR = Path("models")


def main(data_path: Path, models_dir: Path) -> None:
    # Use the file store so CI never needs a pre-existing mlflow.db.
    # This call must come before tune_and_train_model(), which internally calls
    # mlflow.set_tracking_uri("sqlite:///mlflow.db") — setting it here first
    # takes precedence because mlflow.set_tracking_uri is global state and the
    # last call before mlflow.set_experiment() wins.
    mlflow.set_tracking_uri("file:./mlruns")

    # Regenerate processed CSV if absent (e.g. first CI run from clean checkout).
    if not data_path.exists():
        print(f"Processed data not found at {data_path}. Running preprocessing...")
        preprocess_pipeline()
        print("Preprocessing complete.")

    print(f"Training on: {data_path}")
    print(f"Saving models to: {models_dir}\n")

    results = {}
    for model_type in ["logreg", "rf"]:
        print(f"--- Training {model_type} ---")
        _, best_params, cv_metrics, run_id = tune_and_train_model(
            model_type=model_type,
            data_path=data_path,
            models_dir=models_dir,
            experiment_name="heart-disease-classification",
        )
        results[model_type] = {
            "best_params": best_params,
            "best_cv_roc_auc": cv_metrics["best_cv_roc_auc"],
            "run_id": run_id,
        }
        print(f"  Best CV ROC-AUC : {cv_metrics['best_cv_roc_auc']:.4f}")
        print(f"  Best params     : {best_params}")
        print(f"  MLflow run ID   : {run_id}\n")

    # Select champion by highest CV ROC-AUC and write to models/champion.txt.
    # The API reads this file at startup to resolve which .joblib to load —
    # no MLflow server needed at serve time.
    champion = max(results, key=lambda m: results[m]["best_cv_roc_auc"])
    champion_file = models_dir / "champion.txt"
    champion_file.write_text(champion)

    print("=" * 50)
    print("CI Training Summary")
    print("=" * 50)
    for model_type, info in results.items():
        marker = " ✓ champion" if model_type == champion else ""
        print(f"  {model_type:8s}  CV ROC-AUC={info['best_cv_roc_auc']:.4f}  run_id={info['run_id']}{marker}")
    print(f"  Champion written to: {champion_file}")
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
    args = parser.parse_args()

    try:
        main(data_path=args.data_path, models_dir=args.models_dir)
    except Exception as exc:
        print(f"ERROR: CI training failed — {exc}", file=sys.stderr)
        sys.exit(1)
