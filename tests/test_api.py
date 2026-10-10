"""Integration tests for the FastAPI prediction API."""

import os
import unittest
from pathlib import Path

# Set MODEL_PATH before importing the app so the lifespan loads the correct file.
# Path is relative to the Assignment/ working directory where pytest is run from.
os.environ["MODEL_PATH"] = str(Path("models/logreg_model.joblib").resolve())

from fastapi.testclient import TestClient  # noqa: E402
from src.api.app import app  # noqa: E402

VALID_PAYLOAD = {
    "age": 63, "trestbps": 145, "chol": 233, "thalach": 150, "oldpeak": 2.3,
    "sex": 1, "cp": 1, "fbs": 1, "restecg": 2, "exang": 0,
    "slope": 3, "ca": 0, "thal": 6,
}


class TestHeartDiseaseAPI(unittest.TestCase):

    def setUp(self):
        # Use TestClient as a context manager so the lifespan runs (loads the model).
        self._client_ctx = TestClient(app)
        self._client_ctx.__enter__()
        self.client = self._client_ctx

    def tearDown(self):
        self._client_ctx.__exit__(None, None, None)

    def test_health_endpoint(self):
        """GET /health returns 200 with status ok."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("model", body)

    def test_predict_valid_input(self):
        """POST /predict with valid 13-feature payload returns prediction and confidence."""
        response = self.client.post("/predict", json=VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("prediction", body)
        self.assertIn("confidence", body)
        self.assertIn(body["prediction"], [0, 1])
        self.assertGreaterEqual(body["confidence"], 0.0)
        self.assertLessEqual(body["confidence"], 1.0)

    def test_predict_missing_field(self):
        """POST /predict with a missing field returns 422 Unprocessable Entity."""
        incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
        response = self.client.post("/predict", json=incomplete)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
