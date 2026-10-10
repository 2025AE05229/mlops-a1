"""Unit tests for feature engineering, training, and model evaluation pipelines."""

import shutil
import tempfile
import unittest
from pathlib import Path
import joblib
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.build_features import create_preprocessor
from src.models.evaluate import evaluate_pipeline
from src.models.train import build_pipeline, train_and_save_model, tune_and_train_model


class TestModelPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.test_dir)

        # Create balanced dummy tabular dataset matching schema (10 samples)
        self.sample_df = pd.DataFrame({
            "age": [58, 60, 44, 52, 63, 53, 57, 48, 49, 64],
            "sex": [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
            "cp": [1, 4, 2, 3, 4, 3, 2, 4, 3, 1],
            "trestbps": [140.0, 130.0, 120.0, 172.0, 145.0, 128.0, 150.0, 110.0, 130.0, 110.0],
            "chol": [283.0, 206.0, 263.0, 199.0, 233.0, 216.0, 168.0, 229.0, 275.0, 211.0],
            "fbs": [1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            "restecg": [2, 2, 0, 0, 2, 2, 0, 0, 0, 2],
            "thalach": [162.0, 132.0, 173.0, 162.0, 150.0, 115.0, 174.0, 168.0, 139.0, 144.0],
            "exang": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            "oldpeak": [1.0, 2.4, 0.0, 0.5, 2.3, 0.0, 1.6, 1.0, 0.2, 1.8],
            "slope": [1, 2, 1, 1, 3, 1, 1, 3, 1, 2],
            "ca": [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            "thal": [3, 7, 7, 7, 6, 3, 3, 7, 3, 3],
            "target": [0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        })
        self.sample_csv = self.tmp_path / "sample_data.csv"
        self.sample_df.to_csv(self.sample_csv, index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_preprocessor(self):
        preprocessor = create_preprocessor()
        X = self.sample_df.drop(columns=["target"])
        transformed = preprocessor.fit_transform(X)
        self.assertEqual(len(transformed), len(X))
        self.assertGreater(transformed.shape[1], len(X.columns))  # categorical one-hot expanded

    def test_build_and_train_pipeline(self):
        for model_type in ["logreg", "rf"]:
            pipeline = build_pipeline(model_type=model_type)
            self.assertIsInstance(pipeline, Pipeline)

            pipe, X_tr, y_tr, X_te, y_te = train_and_save_model(
                model_type=model_type,
                data_path=self.sample_csv,
                models_dir=self.tmp_path / "models",
            )
            saved_file = self.tmp_path / "models" / f"{model_type}_model.joblib"
            self.assertTrue(saved_file.exists())

            # Load and test prediction on raw DataFrame
            loaded_model = joblib.load(saved_file)
            preds = loaded_model.predict(X_te)
            probs = loaded_model.predict_proba(X_te)
            self.assertEqual(len(preds), len(X_te))
            self.assertEqual(probs.shape, (len(X_te), 2))

    def test_evaluate_pipeline(self):
        pipeline = build_pipeline("logreg")
        X = self.sample_df.drop(columns=["target"])
        y = self.sample_df["target"]
        metrics = evaluate_pipeline(pipeline, X.iloc[:8], y.iloc[:8], X.iloc[8:], y.iloc[8:], cv_folds=2)

        self.assertIn("cv_metrics", metrics)
        self.assertIn("test_metrics", metrics)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            self.assertIn(key, metrics["test_metrics"])


class TestMLflowTracking(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.test_dir)

        # Isolate MLflow tracking to a temp SQLite DB so tests never touch mlruns/
        mlflow.set_tracking_uri(f"sqlite:///{self.tmp_path}/test_mlflow.db")

        # 30-sample dataset: 15 per class — enough for 5-fold stratified CV (min 3/fold per class)
        base = {
            "age": [58, 60, 44, 52, 63, 53, 57, 48, 49, 64, 55, 62, 47, 51, 66],
            "sex": [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
            "cp": [1, 4, 2, 3, 4, 3, 2, 4, 3, 1, 2, 3, 1, 4, 2],
            "trestbps": [140, 130, 120, 172, 145, 128, 150, 110, 130, 110, 135, 125, 140, 160, 120],
            "chol": [283, 206, 263, 199, 233, 216, 168, 229, 275, 211, 240, 195, 310, 220, 180],
            "fbs": [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            "restecg": [2, 2, 0, 0, 2, 2, 0, 0, 0, 2, 0, 2, 0, 2, 0],
            "thalach": [162, 132, 173, 162, 150, 115, 174, 168, 139, 144, 155, 145, 170, 130, 160],
            "exang": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            "oldpeak": [1.0, 2.4, 0.0, 0.5, 2.3, 0.0, 1.6, 1.0, 0.2, 1.8, 0.8, 1.5, 0.3, 2.0, 0.1],
            "slope": [1, 2, 1, 1, 3, 1, 1, 3, 1, 2, 1, 2, 1, 3, 1],
            "ca": [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0],
            "thal": [3, 7, 7, 7, 6, 3, 3, 7, 3, 3, 3, 7, 3, 6, 3],
            "target": [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
        # Duplicate to get 30 rows (15 class-0, 15 class-1) for reliable stratified splits
        doubled = {k: v * 2 for k, v in base.items()}
        self.sample_df = pd.DataFrame(doubled)
        self.sample_csv = self.tmp_path / "sample_data.csv"
        self.sample_df.to_csv(self.sample_csv, index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        # Reset tracking URI back to default so other tests are unaffected
        mlflow.set_tracking_uri(None)

    def test_tune_and_train_logs_mlflow_run(self):
        """tune_and_train_model must create an MLflow run with params and best_cv_roc_auc metric."""
        _, best_params, cv_metrics, run_id = tune_and_train_model(
            model_type="logreg",
            data_path=self.sample_csv,
            models_dir=self.tmp_path / "models",
            experiment_name="test-heart-disease",
        )

        # Run ID is returned and is a non-empty string
        self.assertIsInstance(run_id, str)
        self.assertTrue(len(run_id) > 0)

        # The run exists with the expected metric logged.
        # MLflow sklearn autolog logs the best CV score as 'best_cv_score' (MLflow >= 2.x).
        # Filter to the named parent run only — autolog may create additional child runs.
        runs = mlflow.search_runs(
            experiment_names=["test-heart-disease"],
            filter_string="tags.mlflow.runName = 'logreg'",
        )
        self.assertGreaterEqual(len(runs), 1)
        self.assertIn("metrics.best_cv_score", runs.columns)
        self.assertFalse(runs["metrics.best_cv_score"].isna().any())

        # At least one hyperparameter was logged as a param (autolog prefixes with 'best_')
        self.assertTrue(
            any(col.startswith("params.best_classifier__") for col in runs.columns)
        )

        # Returned cv_metrics dict has best_cv_roc_auc key
        self.assertIn("best_cv_roc_auc", cv_metrics)
        self.assertIsInstance(cv_metrics["best_cv_roc_auc"], float)


if __name__ == "__main__":
    unittest.main()
