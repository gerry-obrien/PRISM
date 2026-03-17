from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]  # services -> app -> backend -> PRISM
DATA_FILE = ROOT_DIR / "data" / "listings_predicted.csv"

BOOL_COLS = ["has_elevator", "is_new_build"]

_df: pd.DataFrame | None = None


def _load() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise RuntimeError("Run python ml/train.py first to generate predictions.")
    df = pd.read_csv(DATA_FILE)
    for col in BOOL_COLS:
        df[col] = df[col].astype(str).str.lower() == "true"
    return df


def get_listings() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = _load()
    return _df
