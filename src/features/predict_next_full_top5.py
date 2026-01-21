import pandas as pd
import numpy as np
from itertools import product
from collections import defaultdict

from src.features.predict_next_full import train_digit_model
from src.features.digit_features import add_digit_features
from src.features.combined_features import build_feature_table

DIGIT_COUNT = 6


def main():
    df = build_feature_table(window=10)
    df = add_digit_features(df)

    feature_cols = [
        col for col in df.columns
        if col not in [
            "draw_no", "draw_date", "city",
            "first_prize", "second_prize_1",
            "second_prize_2", "second_prize_3"
        ]
    ]

    X = df[feature_cols].iloc[:-1]
    X_last = df[feature_cols].iloc[[-1]]

    digit_probs = []

    for i in range(DIGIT_COUNT):
        y = df[f"first_prize_d{i+1}"].shift(-1).dropna()
        model = train_digit_model(X, y)

        # Predict distribution via tree voting
        preds = np.array([tree.predict(X_last)[0] for tree in model.estimators_])
        digits, counts = np.unique(preds.round().astype(int), return_counts=True)

        probs = dict(zip(digits, counts / counts.sum()))
        digit_probs.append(probs)

    # Generate combinations (top-5)
    candidates = []

    for combo in product(*[list(p.keys()) for p in digit_probs]):
        prob = np.prod([digit_probs[i].get(combo[i], 0) for i in range(DIGIT_COUNT)])
        number = int("".join(map(str, combo)))
        candidates.append((number, prob))

    top5 = sorted(candidates, key=lambda x: x[1], reverse=True)[:5]

    print("\n=== TOP 5 FULL NUMBER PREDICTIONS ===")
    for num, prob in top5:
        print(num, f"(prob={prob:.6f})")


if __name__ == "__main__":
    main()
