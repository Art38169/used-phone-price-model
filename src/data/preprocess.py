from pathlib import Path
import pandas as pd


KEEP_COLUMNS = [
    "brand",
    "original_price",
    "age_months",
    "condition",
    "battery_health",
    "screen_cracked",
    "body_damage",
    "repair_history",
    "water_damage",
    "market_demand_score",
    "resale_price",
]


def drop_features(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    return df.drop(columns=columns, errors="ignore")


def remove_iqr_outliers(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    cleaned = df.copy()

    for col in columns:
        Q1 = cleaned[col].quantile(0.25)
        Q3 = cleaned[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        cleaned = cleaned[
            (cleaned[col] >= lower)
            & (cleaned[col] <= upper)
        ]

    return cleaned


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:

    cleaned = df.copy()

    cleaned = cleaned[
        cleaned["resale_price"] > 0
    ]

    cleaned = cleaned[
        cleaned["original_price"] > 0
    ]

    cleaned = cleaned[
        cleaned["battery_health"].between(0, 100)
    ]

    cleaned = cleaned[
        cleaned["age_months"] >= 0
    ]

    return cleaned


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()

    condition_map = {
        "Poor": 0,
        "Fair": 1,
        "Good": 2,
        "Excellent": 3,
    }

    encoded["condition"] = encoded["condition"].map(condition_map)


    encoded = pd.get_dummies(
        encoded,
        columns=[
            "brand"
        ],
        drop_first=True,
    )

    return encoded


def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    df = remove_invalid_rows(df)

    drop_columns = [
        col for col in df.columns
        if col not in KEEP_COLUMNS
    ]

    df = drop_features(df, drop_columns)

    df = remove_iqr_outliers(
        df,
        [
            "original_price",
            "resale_price",
        ],
    )

    # Encode categorical features
    df = encode_features(df)

    return df

