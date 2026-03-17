from pathlib import Path
import json
import pandas as pd
import joblib

ROOT_DIR = Path(__file__).resolve().parents[3]  # services -> app -> backend -> PRISM
MODEL_PATH = ROOT_DIR / "ml" / "artifacts" / "model.pkl"
METRICS_PATH = ROOT_DIR / "ml" / "artifacts" / "metrics.json"

BOOL_COLS = ["has_elevator", "is_new_build"]

_pipeline = None
_metrics: dict | None = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Run python ml/train.py first to train the model.")
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def get_metrics() -> dict:
    global _metrics
    if _metrics is None:
        if not METRICS_PATH.exists():
            raise RuntimeError("Run python ml/train.py first to train the model.")
        with open(METRICS_PATH, "r") as f:
            _metrics = json.load(f)
    return _metrics


def predict(data: dict) -> float:
    pipeline = get_pipeline()
    df = pd.DataFrame([data])
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.lower() == "true").astype(int)
    result = pipeline.predict(df)
    return float(result[0])
