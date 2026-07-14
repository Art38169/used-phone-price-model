from pathlib import Path

import pandas as pd

# used-phone-price-model/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "used_phone_price_prediction_1M.csv"
)


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df
