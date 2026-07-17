import matplotlib.pyplot as plt


def resale_price_distribution(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df["resale_price"], bins=50)
    plt.title("Distribution of Resale Price")
    plt.xlabel("Resale Price")
    plt.ylabel("Frequency")
    plt.show()


def depreciation_distribution(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df["depreciation_pct"], bins=50)
    plt.title("Distribution of Price Depreciation (%)")
    plt.xlabel("Depreciation (%)")
    plt.ylabel("Count")
    plt.show()