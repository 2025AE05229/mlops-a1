"""Champion model packaging and MLflow Model Registry registration.

Reads reports/model_evaluation.json to determine the best model by test ROC-AUC,
loads its joblib artifact, infers a ModelSignature from a data sample, and registers
it to the local MLflow Model Registry as 'HeartDiseaseClassifier'.
"""

from pathlib import Path
import json
import sys
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_DATA_PATH = Path("data/processed/heart_disease_cleaned.csv")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_REPORT_PATH = Path("reports/model_evaluation.json")
EXPERIMENT_NAME = "heart-disease-classification"
REGISTERED_MODEL_NAME = "HeartDiseaseClassifier"


def register_champion_model(
    report_path: Path = DEFAULT_REPORT_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    data_path: Path = DEFAULT_DATA_PATH,
) -> str:
    """Loads the champion pipeline, logs it with a ModelSignature, and registers it.

    Returns the registered model URI (e.g. 'models:/HeartDiseaseClassifier/1').
    """
    report_path = Path(report_path)
    with open(report_path) as f:
        evaluation = json.load(f)

    # Pick the model with the highest test ROC-AUC
    champion_type = max(
        evaluation,
        key=lambda m: evaluation[m]["test_metrics"]["roc_auc"],
    )
    champion_auc = evaluation[champion_type]["test_metrics"]["roc_auc"]
    print(f"Champion model: {champion_type} (test ROC-AUC: {champion_auc})")

    pipeline = joblib.load(Path(models_dir) / f"{champion_type}_model.joblib")

    # Use a small sample to infer the input/output signature
    df = pd.read_csv(data_path)
    X_sample = df.drop(columns=["target"]).head(5)
    y_sample_pred = pipeline.predict(X_sample)
    signature = infer_signature(X_sample, y_sample_pred)

    champion_cv = evaluation[champion_type]["cv_metrics"]
    cv_log = {}
    for metric, stats in champion_cv.items():
        cv_log[f"cv_{metric}_mean"] = stats["mean"]
        cv_log[f"cv_{metric}_std"] = stats["std"]

    champion_test = evaluation[champion_type]["test_metrics"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="champion_registration") as run:
        mlflow.set_tag("champion_model_type", champion_type)
        mlflow.log_metrics({f"test_{k}": v for k, v in champion_test.items()})
        mlflow.log_metrics(cv_log)

        dataset = mlflow.data.from_pandas(
            df,
            source=str(data_path),
            name="heart_disease",
            targets="target",
        )
        mlflow.log_input(dataset, context="training")

        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="champion_pipeline",
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    model_uri = f"models:/{REGISTERED_MODEL_NAME}/1"
    print(f"Registered model URI : {model_uri}")
    print(f"MLflow run ID        : {run.info.run_id}")
    print(f"Model artifact URI   : {model_info.model_uri}")
    return model_uri


if __name__ == "__main__":
    register_champion_model()
