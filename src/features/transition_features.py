import numpy as np
import pandas as pd


def build_last_digit_transition_matrix(series: pd.Series) -> np.ndarray:
    """
    Builds a 10x10 last-digit transition count matrix.
    Each cell [i,j] counts how many times digit j followed digit i.
    """
    digits = series.astype(int) % 10
    matrix = np.zeros((10, 10), dtype=int)
    for i in range(1, len(digits)):
        prev_d = digits.iloc[i - 1]
        curr_d = digits.iloc[i]
        matrix[prev_d, curr_d] += 1
    return matrix


def transition_probability_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Converts a count matrix to a probability matrix
    """
    prob_matrix = matrix.astype(float)
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    prob_matrix = np.divide(prob_matrix, row_sums, where=row_sums != 0)
    return prob_matrix


def add_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds transition features for all prize columns:
    - prev_last_digit
    - last_digit
    - transition_prob
    - transition_surprise (-log(prob))

    Also adds generic columns for first_prize to maintain backward compatibility:
    - last_digit
    - prev_last_digit
    - transition_prob
    - transition_surprise
    """

    df = df.copy()
    prize_cols = ['first_prize', 'second_prize_1', 'second_prize_2', 'second_prize_3']

    for col in prize_cols:
        # Build transition matrix
        count_matrix = build_last_digit_transition_matrix(df[col])
        prob_matrix = transition_probability_matrix(count_matrix)

        # Compute last digit
        last_digits = df[col] % 10
        prev_digits = last_digits.shift(1)

        # Assign per-prize columns
        df[f'{col}_last_digit'] = last_digits
        df[f'{col}_prev_last_digit'] = prev_digits

        # Compute transition probability
        def compute_prob(row):
            if pd.isna(row[f'{col}_prev_last_digit']):
                return np.nan
            return prob_matrix[int(row[f'{col}_prev_last_digit']), int(row[f'{col}_last_digit'])]

        df[f'{col}_transition_prob'] = df.apply(compute_prob, axis=1)

        # Compute transition surprise
        df[f'{col}_transition_surprise'] = -np.log(df[f'{col}_transition_prob'].replace(0, np.nan))

    # --- Generic columns for backward compatibility ---
    df['last_digit'] = df['first_prize_last_digit']
    df['prev_last_digit'] = df['first_prize_prev_last_digit']
    df['transition_prob'] = df['first_prize_transition_prob']
    df['transition_surprise'] = df['first_prize_transition_surprise']

    # Fill NaNs safely
    df['transition_prob'] = df['transition_prob'].fillna(0)
    df['transition_surprise'] = df['transition_surprise'].fillna(0)

    return df
