import os
import pandas as pd
import numpy as np

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

# --- Import data and feature modules ---
from src.data.load import load_raw_data
from src.data.clean import clean_data
from src.features.rolling_features import add_rolling_features
from src.features.transition_features import add_transition_features
from src.features.anomaly_features import add_anomaly_features
from src.features.transition_features import build_last_digit_transition_matrix, transition_probability_matrix



def build_feature_table(window: int = 10) -> pd.DataFrame:
    """
    Builds the full feature table:
    1. Loads and cleans raw data
    2. Adds rolling features
    3. Adds transition-based features
    4. Adds anomaly features
    """
    # 1. Load raw data
    df = load_raw_data()

    # 2. Clean the data
    df = clean_data(df)

    # 3. Add rolling features
    df = add_rolling_features(df, window=window)

    # 4. Add transition features
    df = add_transition_features(df)

    # 5. Add anomaly features
    df = add_anomaly_features(df, window=window)

    # 6. Handle missing values safely
    df['prev_last_digit'] = df['prev_last_digit'].fillna(0)
    df['transition_prob'] = df['transition_prob'].fillna(0)
    df['expected_surprise'] = df['expected_surprise'].fillna(
        df['transition_surprise'].rolling(window, min_periods=1).mean()
    )

    rolling_cols = [col for col in df.columns if 'rolling' in col]
    for col in rolling_cols:
        df[col] = df[col].fillna(df[col].mean())

    df['surprise_residual'] = df['surprise_residual'].fillna(0)
    df['surprise_zscore'] = df['surprise_zscore'].fillna(0)
    df['is_anomaly'] = (df['surprise_zscore'].abs() > 2.0).astype(int)

    # 7. Save feature table
    df.to_csv("outputs/prizebond_features.csv", index=False)

    # 8. Save anomalies
    top_anomalies = df[df['is_anomaly'] == 1].sort_values(
        by='surprise_zscore', key=abs, ascending=False
    )
    top_anomalies.to_csv("outputs/top_anomalies.csv", index=False)

    return df


def predict_next_numbers(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Predict next possible numbers based on:
    1. Transition probabilities of last digits
    2. Rolling features of first_prize
    Returns top N predictions with probabilities.
    """

    df_features = df.copy()

    # Use last 10 draws for prediction
    recent_first = df_features['first_prize'].tail(10)
    last_digit = recent_first.iloc[-1] % 10

    # --- Transition probability for last digits ---
    count_matrix = np.zeros((10, 10), dtype=int)
    digits = df_features['first_prize'] % 10
    for i in range(1, len(digits)):
        prev_d = digits.iloc[i - 1]
        curr_d = digits.iloc[i]
        count_matrix[prev_d, curr_d] += 1

    prob_matrix = count_matrix.astype(float)
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    prob_matrix = np.divide(prob_matrix, row_sums, where=row_sums != 0)

    next_digit_probs = prob_matrix[last_digit]

    # --- Incorporate rolling features (mean + std) for weighting ---
    mean_recent = df_features['rolling_mean'].iloc[-1]
    std_recent = df_features['rolling_std'].iloc[-1]

    # Weighting factor: numbers close to mean get higher probability
    candidate_numbers = np.arange(0, 10)  # last digits 0-9
    distance_weight = np.exp(-np.abs(candidate_numbers - (mean_recent % 10)) / (std_recent + 1e-6))

    # Combined probability: transition probability * distance weight
    combined_probs = next_digit_probs * distance_weight
    combined_probs /= combined_probs.sum()  # normalize

    # Select top N last digits
    top_digits = np.argsort(combined_probs)[::-1][:top_n]
    top_probs = combined_probs[top_digits]

    predictions = pd.DataFrame({
        'predicted_last_digit': top_digits,
        'probability': top_probs
    })

    return predictions

if __name__ == "__main__":
    # Build features
    df = build_feature_table(window=10)

    # Predict next numbers
    prediction = predict_next_numbers(df, top_n=5)
    print("=== Next Prize Bond Predictions ===")
    print(prediction)

    # Print summary checks
    print("\n=== Top 5 Most Unusual Draws ===")
    print(df[df['is_anomaly'] == 1].head())

    print("\n=== Top 5 Rows ===")
    print(df.head())

    print("\n=== Missing Values ===")
    print(df.isnull().sum())
