from .constants import TIME_COLUMNS


def describe_time_columns(df):
    return df[TIME_COLUMNS].describe().T


def plot_time_histograms(df):
    df[TIME_COLUMNS].hist(figsize=(12, 8))


def time_correlations(df):
    cols = TIME_COLUMNS + ["resale_price"]
    return (
        df[cols]
        .corr()["resale_price"]
        .sort_values(ascending=False)
    )