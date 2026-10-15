"""Integration tests for the FastAPI prediction API."""

import os
import unittest
from pathlib import Path

# Set MODEL_PATH and test credentials before importing the app so module-level
# code in app.py picks them up at import time.
os.environ["MODEL_PATH"] = str(Path("models/logreg_model.joblib").resolve())
os.environ["API_USERNAME"] = "testuser"
os.environ["API_PASSWORD"] = "testpass"

from fastapi.testclient import TestClient  # noqa: E402
from src.api.app import app, _API_USERNAME, _API_PASSWORD  # noqa: E402

VALID_PAYLOAD = {
    "age": 63, "trestbps": 145, "chol": 233, "thalach": 150, "oldpeak": 2.3,
    "sex": 1, "cp": 1, "fbs": 1, "restecg": 2, "exang": 0,
    "slope": 3, "ca": 0, "thal": 6,
}

# Credentials that match what the app loaded from env vars at import time.
VALID_AUTH = (_API_USERNAME, _API_PASSWORD)
INVALID_AUTH = ("wrong", "credentials")


class TestHeartDiseaseAPI(unittest.TestCase):

    def setUp(self):
        # Use TestClient as a context manager so the lifespan runs (loads the model).
        self._client_ctx = TestClient(app)
        self._client_ctx.__enter__()
        self.client = self._client_ctx

    def tearDown(self):
        self._client_ctx.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # /health — no auth required
    # ------------------------------------------------------------------

    def test_health_endpoint(self):
        """GET /health returns 200 with status ok (no auth needed)."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("model", body)

    # ------------------------------------------------------------------
    # /predict — functional tests (valid auth)
    # ------------------------------------------------------------------

    def test_predict_valid_input(self):
        """POST /predict with valid credentials and payload returns prediction and confidence."""
        response = self.client.post("/predict", json=VALID_PAYLOAD, auth=VALID_AUTH)
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
        response = self.client.post("/predict", json=incomplete, auth=VALID_AUTH)
        self.assertEqual(response.status_code, 422)

    # ------------------------------------------------------------------
    # /predict — authentication tests
    # ------------------------------------------------------------------

    def test_predict_no_auth_returns_401(self):
        """POST /predict without credentials returns 401 Unauthorized."""
        response = self.client.post("/predict", json=VALID_PAYLOAD)
        self.assertEqual(response.status_code, 401)

    def test_predict_wrong_credentials_returns_401(self):
        """POST /predict with wrong credentials returns 401 Unauthorized."""
        response = self.client.post("/predict", json=VALID_PAYLOAD, auth=INVALID_AUTH)
        self.assertEqual(response.status_code, 401)

    def test_predict_wrong_password_returns_401(self):
        """POST /predict with correct username but wrong password returns 401."""
        response = self.client.post("/predict", json=VALID_PAYLOAD, auth=(_API_USERNAME, "wrongpass"))
        self.assertEqual(response.status_code, 401)

    def test_predict_wrong_username_returns_401(self):
        """POST /predict with wrong username but correct password returns 401."""
        response = self.client.post("/predict", json=VALID_PAYLOAD, auth=("wronguser", _API_PASSWORD))
        self.assertEqual(response.status_code, 401)

    def test_predict_401_has_www_authenticate_header(self):
        """401 response on /predict includes WWW-Authenticate: Basic header."""
        response = self.client.post("/predict", json=VALID_PAYLOAD, auth=INVALID_AUTH)
        self.assertEqual(response.status_code, 401)
        self.assertIn("WWW-Authenticate", response.headers)
        self.assertEqual(response.headers["WWW-Authenticate"], "Basic")

    # ------------------------------------------------------------------
    # /metrics — authentication tests
    # ------------------------------------------------------------------

    def test_metrics_valid_auth_returns_200(self):
        """GET /metrics with valid credentials returns 200 and Prometheus text."""
        response = self.client.get("/metrics", auth=VALID_AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers["content-type"])

    def test_metrics_no_auth_returns_401(self):
        """GET /metrics without credentials returns 401 Unauthorized."""
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 401)

    def test_metrics_wrong_credentials_returns_401(self):
        """GET /metrics with wrong credentials returns 401 Unauthorized."""
        response = self.client.get("/metrics", auth=INVALID_AUTH)
        self.assertEqual(response.status_code, 401)

    def test_metrics_401_has_www_authenticate_header(self):
        """401 response on /metrics includes WWW-Authenticate: Basic header."""
        response = self.client.get("/metrics", auth=INVALID_AUTH)
        self.assertEqual(response.status_code, 401)
        self.assertIn("WWW-Authenticate", response.headers)
        self.assertEqual(response.headers["WWW-Authenticate"], "Basic")


if __name__ == "__main__":
    unittest.main()
