import pandas as pd

DIGIT_COUNT = 6  # 6-digit prize bond numbers

def split_number_into_digits(series: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Splits a numeric series into individual digit columns.
    Example: 871778 → prefix_d1=8, prefix_d2=7, ..., prefix_d6=8
    """
    df_digits = pd.DataFrame()

    str_numbers = series.astype(int).astype(str).str.zfill(DIGIT_COUNT)

    for i in range(DIGIT_COUNT):
        df_digits[f"{prefix}_d{i+1}"] = str_numbers.str[i].astype(int)

    return df_digits


def add_digit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds digit-level columns for all prize numbers.
    """
    df = df.copy()

    prize_cols = [
        "first_prize",
        "second_prize_1",
        "second_prize_2",
        "second_prize_3",
    ]

    for col in prize_cols:
        digits_df = split_number_into_digits(df[col], col)
        df = pd.concat([df, digits_df], axis=1)

    return df
