from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.model.models import BaseModel


RESULT_DIR = Path("results")

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def get_feature_importance(
    model: BaseModel,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract native feature importance from a trained model.
    """

    estimator = model.estimator

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_

    elif hasattr(estimator, "coef_"):
        importance = abs(estimator.coef_)

    else:
        raise ValueError(
            f"{model.name} does not support feature importance."
        )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return importance_df


def save_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
) -> None:
    """
    Save feature importance as CSV.
    """

    path = (
        RESULT_DIR /
        f"{model_name}_importance.csv"
    )

    importance_df.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved feature importance to {path}"
    )


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
) -> None:
    """
    Save Top-10 feature importance plot.
    """

    top = (
        importance_df
        .head(10)
        .iloc[::-1]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.barh(
        top["Feature"],
        top["Importance"],
    )

    plt.xlabel("Importance")
    plt.title(
        f"{model_name} Feature Importance"
    )

    plt.tight_layout()

    path = (
        RESULT_DIR /
        f"{model_name}_importance.png"
    )

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    print(
        f"Saved feature importance plot to {path}"
    )


def plot_shap_summary(
    model: BaseModel,
    X_train: pd.DataFrame,
    model_name: str,
) -> None:
    """
    Generate SHAP summary plot.
    """

    estimator = model.estimator

    X_sample = X_train.sample(
        n=min(5000, len(X_train)),
        random_state=42,
    )

    if hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(
            estimator
        )

    elif hasattr(estimator, "coef_"):
        explainer = shap.LinearExplainer(
            estimator,
            X_sample,
        )

    else:
        print(
            "SHAP is not supported for this model."
        )
        return

    shap_values = explainer(
        X_sample
    )

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
    )

    plt.tight_layout()

    path = (
        RESULT_DIR /
        f"{model_name}_shap.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved SHAP summary plot to {path}"
    )


def interpret_model(
    model: BaseModel,
    X_train: pd.DataFrame,
    model_name: str,
) -> None:
    """
    Run the complete interpretation pipeline.
    """

    print("\n========== Model Interpretation ==========")

    importance_df = get_feature_importance(
        model,
        X_train.columns.tolist(),
    )

    print("\nTop 10 Features")

    print(
        importance_df.head(10)
    )

    save_feature_importance(
        importance_df,
        model_name,
    )

    plot_feature_importance(
        importance_df,
        model_name,
    )

    plot_shap_summary(
        model,
        X_train,
        model_name,
    )