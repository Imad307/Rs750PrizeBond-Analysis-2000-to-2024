import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and normalizes the raw prize bond dataset.
    Returns a new DataFrame (does not modify input).
    """

    df = df.copy()

    # 1. Rename columns to Python-safe names
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
    df["draw_date"] = pd.to_datetime(df["draw_date"], dayfirst=True, errors="coerce")

    # 3. Sort by draw date descending (newest → oldest)
    df = df.sort_values("draw_date", ascending=False).reset_index(drop=True)

    return df
