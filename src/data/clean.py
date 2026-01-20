import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and normalizes the raw prize bond dataset.
    Returns a new DataFrame (does not modify input).
    """

    df = df.copy()

    # 1. Rename columns to python-safe names
    df = df.rename(columns={
        "Draw No.": "draw_no",
        "1st": "first_prize",
        "2nd": "second_prize_1",
        "2nd.1": "second_prize_2",
        "2nd.2": "second_prize_3",
        "Date": "draw_date",
        "City": "city"
    })

    # 2. Convert date to datetime
    df["draw_date"] = pd.to_datetime(df["draw_date"], format="%d-%m-%Y")

    # 3. Sort chronologically (oldest → newest)
    df = df.sort_values("draw_date").reset_index(drop=True)

    return df
