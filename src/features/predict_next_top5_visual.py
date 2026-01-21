# File: src/features/predict_next_top5_visual.py

import pandas as pd
import numpy as np
from src.data.clean import clean_data
from src.features.transition_features import build_last_digit_transition_matrix, transition_probability_matrix

def load_and_prepare(file_path="outputs/prizebond_features.csv"):
    df = pd.read_csv(file_path)
    df = clean_data(df)
    return df

def predict_next_last_digits(df, prize_col):
    """
    Predict top 5 next last digits for a prize column
    """
    digits = df[prize_col] % 10
    last_digit = digits.iloc[-1]

    count_matrix = build_last_digit_transition_matrix(df[prize_col])
    prob_matrix = transition_probability_matrix(count_matrix)

    next_digit_probs = prob_matrix[last_digit]

    # Top 5 digits
    top_digits_idx = np.argsort(next_digit_probs)[::-1][:5]
    top_probs = next_digit_probs[top_digits_idx]

    return list(top_digits_idx), list(top_probs)

def generate_full_candidates(df, prize_col, top_digits):
    """
    Generate full candidate numbers by combining last digits with number base
    """
    last_number = df[prize_col].iloc[-1]
    number_base = (last_number // 10) * 10  # remove last digit

    candidates = [number_base + d for d in top_digits]
    return candidates

def main():
    df = load_and_prepare()

    prize_cols = ['first_prize', 'second_prize_1', 'second_prize_2', 'second_prize_3']

    print("\n=== Next Prize Bond Top 5 Full Number Predictions ===\n")
    for col in prize_cols:
        top_digits, top_probs = predict_next_last_digits(df, col)
        full_candidates = generate_full_candidates(df, col, top_digits)

        result_df = pd.DataFrame({
            'predicted_number': full_candidates,
            'predicted_last_digit': top_digits,
            'probability': top_probs
        })

        print(f"{col} Top 5 Predicted Numbers:")
        print(result_df, "\n")

if __name__ == "__main__":
    main()
