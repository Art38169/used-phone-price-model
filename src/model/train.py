from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.model.models import get_models
from src.model.evaluate import (
    evaluate_model,
    print_metrics,
)
from src.model.interpret import interpret_model


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


def train() -> None:
    df = load_dataset()

    X_train, X_test, y_train, y_test = split_data(df)

    models = get_models()

    results = []

    best_model = None
    best_rmse = float("inf")

    for model in models:

        print(f"\nTraining {model.name}...")

        model.train(
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        print_metrics(
            model.name,
            metrics,
        )

        model.save(
            MODEL_DIR / f"{model.name}.pkl",
        )

        results.append(
            {
                "Model": model.name,
                **metrics,
            }
        )

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_model = model

    print("\n========== Summary ==========")

    results_df = (
        pd.DataFrame(results)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    print(results_df)

    if best_model is None:
        raise RuntimeError("No model was trained.")

    print(f"\nBest model: {best_model.name}")

    interpret_model(
        model=best_model,
        X_train=X_train,
        model_name=best_model.name,
    )