import pandas as pd

from .constants import CONDITION_FEATURES


def condition_price(df):
    return (
        df.groupby("condition")["resale_price"]
        .mean()
        .sort_values(ascending=False)
    )


def condition_summary(df):
    return (
        df.groupby("condition")
        .agg(
            Avg_Battery=("battery_health", "mean"),
            Avg_Age=("age_months", "mean"),
            Cracked_Rate=("screen_cracked", "mean"),
            Damage_Rate=("body_damage", "mean"),
            Water_Damage_Rate=("water_damage", "mean"),
            Repair_Rate=("repair_history", "mean"),
        )
    )


def condition_crosstabs(df):
    for feature in CONDITION_FEATURES:
        print(f"\n=== {feature} ===")
        print(
            pd.crosstab(
                df["condition"],
                df[feature],
                normalize="index",
            )
        )