import pandas as pd


def brand_prices(df):
    return (
        df.groupby("brand")["resale_price"]
        .mean()
        .sort_values(ascending=False)
    )


def model_prices(df):
    return (
        df.groupby("model")["resale_price"]
        .mean()
        .sort_values(ascending=False)
    )


def seller_prices(df):
    return (
        df.groupby("seller_type")["resale_price"]
        .mean()
        .sort_values(ascending=False)
    )


def city_prices(df):
    return (
        df.groupby("city_tier")["resale_price"]
        .mean()
        .sort_values(ascending=False)
    )


def os_prices(df):
    return (
        df.groupby("os_type")["resale_price"]
        .mean()
        .sort_values(ascending=False)
    )


def brand_os_table(df):
    return pd.crosstab(df["brand"], df["os_type"])


def brand_os_summary(df):
    return (
        df.groupby(["brand", "os_type"])
        .agg(
            Models=("model", "nunique"),
            Phones=("model", "count"),
        )
    )


def list_models(df):
    for brand in sorted(df["brand"].unique()):
        print(f"\n{brand}")
        print(
            sorted(
                df[df["brand"] == brand]["model"].unique()
            )
        )


def apple_models(df):
    return (
        df[df["brand"] == "Apple"]
        .groupby("model")["resale_price"]
        .mean()
        .sort_values(ascending=False)
    )