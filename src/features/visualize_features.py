# src/features/visualize_features.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots look nicer
sns.set(style="whitegrid")

def main():
    # Load the combined feature table
    df = pd.read_csv("outputs/prizebond_features.csv")
    print("=== Top 5 Rows ===")
    print(df.head())

    # Ensure Draw No. is sorted descending (latest draw first)
    df = df.sort_values("Draw No.", ascending=False)

    # ===============================
    # 1. Rolling Mean Plot
    # ===============================
    plt.figure(figsize=(12, 6))
    plt.plot(df["Draw No."], df["rolling_mean_10"], marker='o', linestyle='-', label="Rolling Mean 10")
    plt.gca().invert_xaxis()
    plt.title("Rolling Mean (Window=10) Across Draws")
    plt.xlabel("Draw No.")
    plt.ylabel("Rolling Mean")
    plt.legend()
    plt.show()

    # ===============================
    # 2. Rolling Mean with Anomalies
    # ===============================
    plt.figure(figsize=(12, 6))
    plt.plot(df["Draw No."], df["rolling_mean_10"], marker='o', linestyle='-', label="Rolling Mean 10")
    anomalies = df[df["is_anomaly"] == 1]
    plt.scatter(anomalies["Draw No."], anomalies["rolling_mean_10"], color='red', s=100, label="Anomaly")
    plt.gca().invert_xaxis()
    plt.title("Rolling Mean with Anomalies Highlighted")
    plt.xlabel("Draw No.")
    plt.ylabel("Rolling Mean")
    plt.legend()
    plt.show()

    # ===============================
    # 3. Distribution of Transition Surprise
    # ===============================
    plt.figure(figsize=(10, 5))
    sns.histplot(df["transition_surprise"], bins=20, kde=True)
    plt.title("Distribution of Transition Surprise")
    plt.xlabel("Transition Surprise")
    plt.ylabel("Frequency")
    plt.show()

    # ===============================
    # 4. Correlation Heatmap (numeric only)
    # ===============================
    plt.figure(figsize=(12, 8))

    # Select only numeric columns
    numeric_df = df.select_dtypes(include='number')

    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap (Numeric Columns Only)")
    plt.show()

    # ===============================
    # 5. Summary Statistics
    # ===============================
    print("\n=== Summary Statistics ===")
    print(df.describe())

    print("\n=== Missing Values ===")
    print(df.isna().sum())

if __name__ == "__main__":
    main()