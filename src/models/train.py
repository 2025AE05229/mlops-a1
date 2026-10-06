"""Model architecture, training pipeline, hyperparameter tuning, and artifact saving."""

from pathlib import Path
import sys
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.build_features import create_preprocessor

DEFAULT_DATA_PATH = Path("data/processed/heart_disease_cleaned.csv")
DEFAULT_MODELS_DIR = Path("models")


def build_pipeline(model_type: str = "rf", **hyperparams) -> Pipeline:
    """Builds an end-to-end Pipeline with preprocessing and classifier."""
    preprocessor = create_preprocessor()

    if model_type == "logreg":
        params = {"C": 1.0, "max_iter": 1000, "random_state": 42}
        params.update(hyperparams)
        clf = LogisticRegression(**params)
    elif model_type == "rf":
        params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
        params.update(hyperparams)
        clf = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def tune_and_train_model(
    model_type: str = "rf",
    data_path: Path = DEFAULT_DATA_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    experiment_name: str = "heart-disease-classification",
) -> tuple[Pipeline, dict, dict, str]:
    """Tunes hyperparameters via Stratified 5-Fold GridSearch, saves best pipeline and returns metrics.

    Opens an MLflow run under `experiment_name`, logs hyperparameters, best CV ROC-AUC, and the
    fitted pipeline as an MLflow sklearn artifact. Returns the active run_id so callers can resume
    the same run to log additional metrics and artifacts.
    """
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_pipe = build_pipeline(model_type=model_type)

    if model_type == "logreg":
        param_grid = {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__solver": ["lbfgs", "liblinear"],
        }
    elif model_type == "rf":
        param_grid = {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [3, 5, 8, None],
            "classifier__min_samples_split": [2, 5],
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    # autolog captures: best_params, best_cv_roc_auc, model artifact, CV results table.
    # max_tuning_runs=0 keeps everything on the parent run instead of spawning child runs
    # for every GridSearch candidate.
    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=True,
        log_models=True,
        max_tuning_runs=0,
        silent=True,
    )

    with mlflow.start_run(run_name=model_type) as run:
        grid_search.fit(X_train, y_train)  # autolog fires here

        # dataset inputs are not captured by autolog — log them manually
        train_dataset = mlflow.data.from_pandas(
            pd.concat([X_train, y_train], axis=1),
            source=str(data_path),
            name="heart_disease_train",
            targets="target",
        )
        test_dataset = mlflow.data.from_pandas(
            pd.concat([X_test, y_test], axis=1),
            source=str(data_path),
            name="heart_disease_test",
            targets="target",
        )
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(test_dataset, context="validation")

        run_id = run.info.run_id

    mlflow.sklearn.autolog(disable=True)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_auc = float(grid_search.best_score_)

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_file = models_dir / f"{model_type}_model.joblib"
    joblib.dump(best_pipeline, out_file)
    print(f"Saved tuned {model_type} pipeline to {out_file} (Best CV ROC-AUC: {best_cv_auc:.4f})")

    return best_pipeline, best_params, {"best_cv_roc_auc": round(best_cv_auc, 4)}, run_id


def train_and_save_model(
    model_type: str = "rf",
    data_path: Path = DEFAULT_DATA_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    **hyperparams,
) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Directly trains pipeline with specified hyperparams and saves artifact."""
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(model_type=model_type, **hyperparams)
    pipeline.fit(X_train, y_train)

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_file = models_dir / f"{model_type}_model.joblib"
    joblib.dump(pipeline, out_file)

    return pipeline, X_train, y_train, X_test, y_test


if __name__ == "__main__":
    tune_and_train_model("logreg")
    tune_and_train_model("rf")
