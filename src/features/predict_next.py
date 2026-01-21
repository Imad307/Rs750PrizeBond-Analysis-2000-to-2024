# File: src/features/predict_next.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def load_features(file_path="outputs/prizebond_features.csv"):
    """
    Load combined features CSV with updated rolling and transition features
    """
    df = pd.read_csv(file_path)
    return df


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create next draw columns as target
    """
    df['next_first_prize'] = df['first_prize'].shift(-1)
    df['next_second_prize_1'] = df['second_prize_1'].shift(-1)
    df['next_second_prize_2'] = df['second_prize_2'].shift(-1)
    df['next_second_prize_3'] = df['second_prize_3'].shift(-1)

    # Drop last row which will have NaN
    df = df.dropna(subset=['next_first_prize', 'next_second_prize_1',
                           'next_second_prize_2', 'next_second_prize_3'])
    return df


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """
    Train Random Forest Regressor
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f}")
    return model


def build_features_for_prize(df: pd.DataFrame, prize_col: str, window: int = 10) -> list:
    """
    Returns a list of feature column names for a given prize
    """
    feature_list = [
        f"{prize_col}_prev_last_digit",
        f"{prize_col}_last_digit",
        f"{prize_col}_transition_prob",
        f"{prize_col}_transition_surprise",
        f"rolling_mean_{prize_col}_{window}",
        f"rolling_std_{prize_col}_{window}",
        f"rolling_entropy_{prize_col}_{window}",
        f"rolling_runs_{prize_col}_{window}",
        f"rolling_digit_dominance_{prize_col}_{window}",
        "is_anomaly"
    ]
    return feature_list


def main():
    # Step 1: Load features
    df = load_features()

    # Step 2: Prepare next draw targets
    df = prepare_target(df)

    # Step 3: Train and predict for each prize
    prize_columns = [
        ('first_prize', 'next_first_prize'),
        ('second_prize_1', 'next_second_prize_1'),
        ('second_prize_2', 'next_second_prize_2'),
        ('second_prize_3', 'next_second_prize_3')
    ]

    predictions = {}

    for prize_col, target_col in prize_columns:
        features = build_features_for_prize(df, prize_col)
        X = df[features]
        y = df[target_col]

        print(f"\nTraining model for {prize_col}...")
        model = train_model(X, y)

        # Predict next number using last row of features
        X_last = X.iloc[[-1]]
        pred = round(model.predict(X_last)[0])
        predictions[prize_col] = pred

    # Step 4: Print predictions
    print("\n=== Predicted Next Draw Numbers ===")
    for prize, value in predictions.items():
        print(f"{prize}: {value}")


if __name__ == "__main__":
    main()
