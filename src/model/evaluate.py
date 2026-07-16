from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.model.model_all import BaseModel


def evaluate_model(
    model: BaseModel,
    X_test,
    y_test,
) -> dict[str, float]:
    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def print_metrics(
    model_name: str,
    metrics: dict[str, float],
) -> None:
    
    print(f"\n===== {model_name} =====")
    print(f"MAE  : {metrics['MAE']:.2f}")
    print(f"RMSE : {metrics['RMSE']:.2f}")
    print(f"R²   : {metrics['R2']:.4f}")