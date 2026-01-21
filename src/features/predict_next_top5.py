# File: src/features/predict_next_top5.py

import pandas as pd
import numpy as np
from src.data.clean import clean_data
from src.features.rolling_features import add_rolling_features
from src.features.transition_features import add_transition_features, build_last_digit_transition_matrix, transition_probability_matrix

def load_and_prepare(file_path="outputs/prizebond_features.csv"):
    """
    Load feature CSV, clean, and add rolling & transition features
    """
    df = pd.read_csv(file_path)

    # Clean data
    df = clean_data(df)

    # Add rolling features
    df = add_rolling_features(df, window=10)

    # Add transition features for all prize columns
    df = add_transition_features(df)

    return df

def predict_top5_last_digits(df: pd.DataFrame, prize_col="first_prize", top_n=5):
    """
    Predict Top N most likely last digits for the next draw of a prize column
    """
    # Last row
    last_row = df.iloc[-1]

    # Build transition probability matrix
    count_matrix = build_last_digit_transition_matrix(df[prize_col])
    prob_matrix = transition_probability_matrix(count_matrix)

    # Last observed digit
    last_digit = last_row[f"{prize_col}_last_digit"]

    # Next digit probabilities
    next_digit_probs = prob_matrix[int(last_digit)]

    # Get top N digits
    top_digits = np.argsort(next_digit_probs)[::-1][:top_n]
    top_probs = next_digit_probs[top_digits]

    predictions = pd.DataFrame({
        "predicted_last_digit": top_digits,
        "probability": top_probs
    })

    return predictions

def main():
    df = load_and_prepare()

    prize_columns = ["first_prize", "second_prize_1", "second_prize_2", "second_prize_3"]

    print("=== Next Prize Bond Top 5 Predictions ===")
    for col in prize_columns:
        top5 = predict_top5_last_digits(df, prize_col=col, top_n=5)
        print(f"\n{col} Top 5 Predicted Last Digits:")
        print(top5)

if __name__ == "__main__":
    main()
