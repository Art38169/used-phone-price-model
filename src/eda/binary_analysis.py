import pandas as pd

from .constants import BINARY_COLUMNS


def binary_feature_summary(df):
    summary = []

    for col in BINARY_COLUMNS:
        avg = df.groupby(col)["resale_price"].mean()

        summary.append(
            {
                "Feature": col,
                "Mean (0)": avg.loc[0],
                "Mean (1)": avg.loc[1],
                "% Change": (
                    (avg.loc[1] - avg.loc[0])
                    / avg.loc[0]
                ) * 100,
            }
        )

    return pd.DataFrame(summary)