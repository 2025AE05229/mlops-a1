"""FastAPI application for Heart Disease prediction.

Loads the champion sklearn Pipeline at startup and exposes:
  GET  /health   — liveness check
  POST /predict  — returns binary prediction and confidence score
  GET  /metrics  — Prometheus-format metrics (request counts, latency histograms)

Champion resolution order (first match wins):
  1. MODEL_PATH env var — explicit override, e.g. for testing a specific model
  2. models/champion.txt — written by ci_train.py after each training run;
     contains the model type name ("logreg" or "rf")
  3. Fallback default: "logreg"

Environment variables:
  MODEL_PATH  explicit path to a .joblib file (overrides champion.txt)
  PORT        port uvicorn listens on (default: 8080, overridden by IBM Code Engine)
"""

import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.ml_pipeline.tasks.build_features import CATEGORICAL_COLS, NUMERICAL_COLS

# ---------------------------------------------------------------------------
# Structured JSON logger
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("api")

# Column order must match the order the Pipeline was trained on
FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

# ---------------------------------------------------------------------------
# Basic Authentication
# ---------------------------------------------------------------------------

_security = HTTPBasic()

_API_USERNAME = os.getenv("API_USERNAME", "admin")
_API_PASSWORD = os.getenv("API_PASSWORD", "7fc8da10-7191-4c71-af80-ad59ba526291")


def _require_auth(credentials: HTTPBasicCredentials = Depends(_security)):
    """Verify HTTP Basic credentials against API_USERNAME / API_PASSWORD env vars.

    Uses secrets.compare_digest to prevent timing attacks.
    Raises HTTP 401 on invalid credentials.
    """
    valid_user = secrets.compare_digest(credentials.username.encode(), _API_USERNAME.encode())
    valid_pass = secrets.compare_digest(credentials.password.encode(), _API_PASSWORD.encode())
    if not (valid_user and valid_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
CHAMPION_FILE = MODELS_DIR / "champion.txt"
_FALLBACK_MODEL_TYPE = "logreg"


def _resolve_model_path() -> Path:
    """Determine which .joblib to load.

    Priority:
      1. MODEL_PATH env var (explicit override)
      2. models/champion.txt written by ci_train.py
      3. Hardcoded fallback to logreg
    """
    explicit = os.getenv("MODEL_PATH")
    if explicit:
        return Path(explicit)
    if CHAMPION_FILE.exists():
        model_type = CHAMPION_FILE.read_text().strip()
        return MODELS_DIR / f"{model_type}_model.joblib"
    return MODELS_DIR / f"{_FALLBACK_MODEL_TYPE}_model.joblib"


MODEL_PATH = _resolve_model_path()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PatientFeatures(BaseModel):
    # Numerical features
    age: float
    trestbps: float
    chol: float
    thalach: float
    oldpeak: float
    # Categorical features (encoded as integers in the dataset)
    sex: int
    cp: int
    fbs: int
    restecg: int
    exang: int
    slope: int
    ca: int
    thal: int


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup; release on shutdown."""
    app.state.model = joblib.load(MODEL_PATH)
    yield
    del app.state.model


app = FastAPI(
    title="Heart Disease Classifier API",
    description="Predicts risk of heart disease from patient health data.",
    version="1.0.0",
    lifespan=lifespan,
)

# Instrument all routes; DO NOT auto-expose /metrics — we add it below with auth.
Instrumentator().instrument(app)


# ---------------------------------------------------------------------------
# Middleware — structured request logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request as a single JSON line to stdout."""
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": latency_ms,
    }))
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check — returns service status and resolved model name."""
    return {"status": "ok", "model": MODEL_PATH.stem.replace("_model", "")}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures, _: None = Depends(_require_auth)):
    """Run inference on a single patient record.

    Returns:
        prediction  — 0 (no disease) or 1 (disease present)
        confidence  — probability of class 1 (disease present), range [0, 1]
    """
    # Build single-row DataFrame in the exact column order the Pipeline expects
    row = {col: getattr(features, col) for col in FEATURE_COLS}
    X = pd.DataFrame([row])

    model = app.state.model
    prediction = int(model.predict(X)[0])
    confidence = float(model.predict_proba(X)[0, 1])

    logger.info(json.dumps({
        "event": "prediction",
        "prediction": prediction,
        "confidence": round(confidence, 4),
    }))

    return PredictionResponse(prediction=prediction, confidence=confidence)


@app.get("/metrics")
def metrics(_: None = Depends(_require_auth)):
    """Prometheus metrics — protected by HTTP Basic Authentication."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
