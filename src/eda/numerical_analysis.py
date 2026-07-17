import matplotlib.pyplot as plt

from .constants import HARDWARE_COLUMNS


def original_vs_resale(df):
    sample = df.sample(10000, random_state=42)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        sample["original_price"],
        sample["resale_price"],
        alpha=0.5,
    )
    plt.xlabel("Original Price")
    plt.ylabel("Resale Price")
    plt.show()


def age_vs_resale(df):
    sample = df.sample(10000, random_state=42)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        sample["age_months"],
        sample["resale_price"],
        alpha=0.3,
    )
    plt.xlabel("Age")
    plt.ylabel("Resale Price")
    plt.show()


def age_vs_depreciation(df):
    sample = df.sample(10000, random_state=42)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        sample["age_months"],
        sample["depreciation_pct"],
        alpha=0.3,
    )
    plt.xlabel("Age")
    plt.ylabel("Depreciation (%)")
    plt.show()


def battery_health_analysis(df):
    avg = (
        df.groupby("battery_health")["resale_price"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(avg.index, avg.values)
    plt.show()

    return avg


def hardware_correlations(df):
    return df[HARDWARE_COLUMNS].corr()