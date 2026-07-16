from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.model.evaluate import (
    evaluate_model,
    print_metrics,
)


MODEL_DIR = Path("models")


def load_dataset() -> pd.DataFrame:
    df = load_data()

    return preprocess(df)


def split_data(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    X = df.drop(columns=["resale_price"])

    y = df["resale_price"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


def get_models() -> dict:
    return {
        "linear_regression": LinearRegression(),

        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),

        "xgboost": XGBRegressor(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
        ),
        "lightgbm": LGBMRegressor(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
        )
    }


def train_model(
    model,
    X_train,
    y_train,
):
    model.fit(
        X_train,
        y_train,
    )

    return model


def save_model(
    model,
    filename: str,
) -> None:

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = MODEL_DIR / filename

    joblib.dump(
        model,
        path,
    )

    print(f"Saved model to {path}")


def train() -> None:
    df = load_dataset()

    X_train, X_test, y_train, y_test = split_data(df)

    models = get_models()

    results = []

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        trained_model = train_model(
            model,
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            trained_model,
            X_test,
            y_test,
        )

        print_metrics(
            model_name,
            metrics,
        )

        save_model(
            trained_model,
            f"{model_name}.pkl",
        )

        results.append({
            "Model": model_name,
            **metrics,
        })

    print("\n========== Summary ==========")

    results_df = (
        pd.DataFrame(results)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    print(results_df)

