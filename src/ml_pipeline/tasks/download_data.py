"""Data acquisition script for the UCI Heart Disease dataset."""

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DEFAULT_OUTPUT = Path("data/raw/processed.cleveland.data")
DEFAULT_FALLBACK = Path("heart+disease/processed.cleveland.data")


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def acquire_data(output_path: Path = DEFAULT_OUTPUT, fallback_path: Path = DEFAULT_FALLBACK) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    downloaded = False
    try:
        print(f"Downloading from {UCI_URL}...")
        urllib.request.urlretrieve(UCI_URL, out)
        downloaded = True
    except Exception as e:
        print(f"Download failed ({e}). Attempting fallback...")

    if not downloaded or not out.exists() or out.stat().st_size == 0:
        if fallback_path and os.path.exists(fallback_path):
            print(f"Copying from local fallback: {fallback_path}")
            shutil.copyfile(fallback_path, out)
        else:
            raise FileNotFoundError(f"Could not acquire dataset from URL or fallback path: {fallback_path}")

    print(f"Acquired: {out} ({out.stat().st_size} bytes, sha256={compute_sha256(out)[:10]})")
    return out


if __name__ == "__main__":
    acquire_data()
