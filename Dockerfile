FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

# Install runtime-only dependencies (no mlflow, matplotlib, pytest, etc.)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy application source and all model artifacts (both .joblib files + champion.txt)
COPY src/ ./src/
COPY models/ ./models/

# Default environment — overridden by IBM Code Engine at runtime
ENV PORT=8080

EXPOSE 8080

# Use sh -c so ${PORT} is expanded from the runtime environment variable.
# IBM Code Engine injects PORT at deploy time; the default above is used locally.
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}"]
