
def add_anomaly_features(df, window=10):
    """
    Detects local anomalies using transition surprise residuals.
    """
    df = df.copy()

    # Rolling expectation of surprise
    df["expected_surprise"] = df["transition_surprise"].rolling(window).mean()

    # Residual = actual - expected
    df["surprise_residual"] = df["transition_surprise"] - df["expected_surprise"]

    # Rolling std for normalization
    rolling_std = df["transition_surprise"].rolling(window).std()

    # Z-score (how extreme is this draw locally?)
    df["surprise_zscore"] = df["surprise_residual"] / rolling_std

    # Anomaly flag (tunable threshold)
    df["is_anomaly"] = (df["surprise_zscore"].abs() > 2.0).astype(int)

    return df


if __name__ == "__main__":
    # Absolute imports based on your project structure
    from src.data.load import load_raw_data
    from src.features.transition_features import add_transition_features

    # Load raw data
    df = load_raw_data()

    # Step 6.2: Add transition-based features
    df = add_transition_features(df)

    # Step 6.3: Add anomaly features
    df = add_anomaly_features(df, window=10)

    # Display the last 20 rows with anomaly info
    print(df[[
        "Draw No.",
        "transition_surprise",
        "surprise_residual",
        "surprise_zscore",
        "is_anomaly"
    ]].tail(20))
