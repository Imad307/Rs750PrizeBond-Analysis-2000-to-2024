import numpy as np
import pandas as pd
from scipy.stats import entropy


def last_digit_entropy(series: pd.Series) -> float:
    """Entropy of last digits in a window"""
    digits = series.astype(int) % 10
    counts = np.bincount(digits, minlength=10)
    probs = counts / counts.sum() if counts.sum() > 0 else np.zeros(10)
    return entropy(probs)


def runs_count(series: pd.Series) -> int:
    """Count runs (clusters) in a numeric sequence"""
    if len(series) < 2:
        return 0
    runs = 1
    for i in range(1, len(series)):
        if series.iloc[i] != series.iloc[i - 1]:
            runs += 1
    return runs


def digit_dominance(series: pd.Series) -> float:
    """Measures dominance of most frequent last digit"""
    digits = series.astype(int) % 10
    counts = np.bincount(digits, minlength=10)
    return counts.max() - counts.mean()


def add_rolling_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Adds rolling features for all prize columns:
    - Rolling mean
    - Rolling standard deviation
    - Last-digit entropy
    - Runs count
    - Digit dominance

    Also adds generic columns for first_prize to maintain backward compatibility:
    - rolling_mean
    - rolling_std
    - rolling_entropy
    - rolling_runs
    - rolling_digit_dominance
    """

    df = df.copy()
    prize_cols = ['first_prize', 'second_prize_1', 'second_prize_2', 'second_prize_3']

    for col in prize_cols:
        # Rolling mean
        df[f'rolling_mean_{col}_{window}'] = df[col].rolling(window, min_periods=1).mean()
        # Rolling standard deviation
        df[f'rolling_std_{col}_{window}'] = df[col].rolling(window, min_periods=1).std().fillna(0)
        # Last-digit entropy
        df[f'rolling_entropy_{col}_{window}'] = df[col].rolling(window, min_periods=1).apply(
            last_digit_entropy, raw=False
        )
        # Runs count
        df[f'rolling_runs_{col}_{window}'] = df[col].rolling(window, min_periods=1).apply(
            runs_count, raw=False
        )
        # Digit dominance
        df[f'rolling_digit_dominance_{col}_{window}'] = df[col].rolling(window, min_periods=1).apply(
            digit_dominance, raw=False
        )

    # --- Generic columns for backward compatibility ---
    df['rolling_mean'] = df[f'rolling_mean_first_prize_{window}']
    df['rolling_std'] = df[f'rolling_std_first_prize_{window}']
    df['rolling_entropy'] = df[f'rolling_entropy_first_prize_{window}']
    df['rolling_runs'] = df[f'rolling_runs_first_prize_{window}']
    df['rolling_digit_dominance'] = df[f'rolling_digit_dominance_first_prize_{window}']

    # Fill NaNs safely
    df['rolling_mean'] = df['rolling_mean'].fillna(0)
    df['rolling_std'] = df['rolling_std'].fillna(0)
    df['rolling_entropy'] = df['rolling_entropy'].fillna(0)
    df['rolling_runs'] = df['rolling_runs'].fillna(0)
    df['rolling_digit_dominance'] = df['rolling_digit_dominance'].fillna(0)

    return df
