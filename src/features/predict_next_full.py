import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.data.clean import clean_data
from src.features.combined_features import build_feature_table
from src.features.digit_features import add_digit_features

DIGIT_COUNT = 6


def train_digit_model(X, y):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def main():
    # 1. Build full feature table
    df = build_feature_table(window=10)

    # 2. Add digit features
    df = add_digit_features(df)

    # 3. Build digit targets (NEXT draw)
    models = {}

    feature_cols = [
        col for col in df.columns
        if col not in [
            "draw_no", "draw_date", "city",
            "first_prize", "second_prize_1",
            "second_prize_2", "second_prize_3"
        ]
    ]

    X = df[feature_cols].iloc[:-1]  # drop last row
    X_last = df[feature_cols].iloc[[-1]]  # for prediction

    print("\n=== Training digit models for FIRST PRIZE ===")

    predicted_digits = []

    for i in range(DIGIT_COUNT):
        target_col = f"first_prize_d{i+1}"
        y = df[target_col].shift(-1).dropna()

        model = train_digit_model(X, y)
        models[target_col] = model

        digit_pred = int(round(model.predict(X_last)[0]))
        digit_pred = max(0, min(9, digit_pred))  # clamp

        predicted_digits.append(str(digit_pred))

    predicted_number = int("".join(predicted_digits))

    print("\n=== Predicted NEXT FIRST PRIZE NUMBER ===")
    print(predicted_number)


if __name__ == "__main__":
    main()
