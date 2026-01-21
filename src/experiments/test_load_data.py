from src.data.load import load_raw_data
from src.data.validate import validate_schema
from src.data.clean import clean_data
from src.data.save import save_clean_data


def main():
    df_raw = load_raw_data()
    validate_schema(df_raw)

    df_clean = clean_data(df_raw)
    save_clean_data(df_clean)

    print("Cleaned data saved successfully")
    print(df_clean.head())
    print(df_clean.tail())
    print(df_clean.shape)


if __name__ == "__main__":
    main()
