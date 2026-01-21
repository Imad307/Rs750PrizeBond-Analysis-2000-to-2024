from src.data.load import load_raw_data
from src.features.rolling_features import add_rolling_features


def main():
    df = load_raw_data()
    df = add_rolling_features(df, column="1st")

    print(df.head(15))
    print(df.columns)


if __name__ == "__main__":
    main()
