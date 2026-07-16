from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap


RESULT_DIR = Path("results")
RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def get_feature_importance(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Return native feature importance.
    """

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    elif hasattr(model, "coef_"):
        importance = abs(model.coef_)

    else:
        raise ValueError(
            "Model does not support feature importance."
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

    csv_path = (
        RESULT_DIR /
        f"{model_name}_importance.csv"
    )

    importance_df.to_csv(
        csv_path,
        index=False,
    )

    print(
        f"Saved feature importance to {csv_path}"
    )


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
) -> None:

    top = (
        importance_df
        .head(10)
        .iloc[::-1]
    )

    plt.figure(figsize=(8, 5))

    plt.barh(
        top["Feature"],
        top["Importance"],
    )

    plt.xlabel("Importance")
    plt.title(
        f"{model_name} Feature Importance"
    )

    plt.tight_layout()

    plot_path = (
        RESULT_DIR /
        f"{model_name}_importance.png"
    )

    plt.savefig(plot_path)

    plt.close()

    print(
        f"Saved feature importance plot to {plot_path}"
    )


def plot_shap_summary(
    model,
    X_train: pd.DataFrame,
    model_name: str,
) -> None:

    X_sample = X_train.sample(
        n=min(5000, len(X_train)),
        random_state=42,
    )

    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(
            model,
            X_sample,
        )

    shap_values = explainer(X_sample)

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
    )

    plt.tight_layout()

    shap_path = (
        RESULT_DIR /
        f"{model_name}_shap.png"
    )

    plt.savefig(
        shap_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved SHAP summary plot to {shap_path}"
    )


def interpret_model(
    model,
    X_train: pd.DataFrame,
    model_name: str,
) -> None:

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