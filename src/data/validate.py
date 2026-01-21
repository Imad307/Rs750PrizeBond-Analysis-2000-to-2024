import pandas as pd


REQUIRED_COLUMNS = {
    "Draw No.",
    "1st",
    "2nd",
    "2nd.1",
    "2nd.2",
    "City",
    "Date",
}


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validates the raw dataset schema.
    Raises AssertionError if schema is invalid.
    """

    # 1. Check required columns
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    assert not missing_cols, f"Missing columns: {missing_cols}"

    # 2. Check row count sanity
    assert len(df) > 0, "Dataset is empty"

    # 3. Each draw must have exactly 4 winning numbers
    prize_columns = ["1st", "2nd", "2nd.1", "2nd.2"]
    assert df[prize_columns].notnull().all().all(), \
        "Missing prize numbers detected"

    # 4. Draw numbers must be unique
    assert df["Draw No."].is_unique, "Duplicate draw numbers found"

    # 5. Winning numbers must be integers
    for col in prize_columns:
        assert pd.api.types.is_integer_dtype(df[col]), \
            f"Column {col} must be integer type"
