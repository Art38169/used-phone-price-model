def resale_price_correlations(df):
    return (
        df.corr(numeric_only=True)["resale_price"]
        .sort_values(ascending=False)
    )


def add_depreciation_column(df):
    df = df.copy()

    df["depreciation_pct"] = (
        (df["original_price"] - df["resale_price"])
        / df["original_price"]
    ) * 100

    return df