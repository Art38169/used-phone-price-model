from src.data.load_data import load_data
from src.data.preprocess import preprocess


def main():
    df = load_data()

    print("Before preprocessing:")
    print(df.shape)

    df = preprocess(df)

    print("After preprocessing:")
    print(df.shape)

    print(df.head())


if __name__ == "__main__":
    main()