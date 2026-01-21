import numpy as np
import pandas as pd
from statsmodels.sandbox.stats.runs import runstest_1samp
from src.data.load import load_raw_data
from src.data.validate import validate_schema

def main():
    df = load_raw_data()
    validate_schema(df)
    prize_cols = ["1st", "2nd", "2nd.1", "2nd.2"]
    city_results = []

    for city, city_df in df.groupby("City"):
        numbers = city_df[prize_cols].values.flatten()
        numbers = numbers[~np.isnan(numbers)]

        z_stat, p_value = runstest_1samp(numbers, correction=False)

        city_results.append({
            "City": city,
            "Draws": len(city_df),
            "Mean": np.mean(numbers),
            "Std_Dev": np.std(numbers),
            "Runs_Test_p": p_value
        })

    city_analysis_df = pd.DataFrame(city_results)
    city_analysis_df = city_analysis_df.sort_values("Runs_Test_p")

    print("\nCity-wise Randomness Analysis:\n")
    print(city_analysis_df)


if __name__ == "__main__":
    main()
