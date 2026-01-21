import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load features
df = pd.read_csv("../outputs/prizebond_features.csv")

# Quick summary
print("=== Summary Statistics ===")
print(df.describe())
print("\n=== Missing Values ===")
print(df.isna().sum())

# Plot anomalies over draws
plt.figure(figsize=(12,5))
plt.plot(df["Draw No."], df["surprise_zscore"], label="Surprise Z-Score")
plt.scatter(df["Draw No."], df["surprise_zscore"], c=df["is_anomaly"], cmap='coolwarm', s=50, label="Anomaly")
plt.gca().invert_xaxis()
plt.xlabel("Draw No.")
plt.ylabel("Surprise Z-Score")
plt.title("Anomalies in Prize Bond Draws")
plt.legend()
plt.show()

# Correlation heatmap
plt.figure(figsize=(15,12))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

# Distribution of rolling and transition features
rolling_cols = [col for col in df.columns if "rolling" in col]
transition_cols = ["transition_prob", "transition_surprise"]

plt.figure(figsize=(15,10))
for i, col in enumerate(rolling_cols + transition_cols, 1):
    plt.subplot(4, 4, i)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()
