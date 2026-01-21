import pandas as pd
import numpy as np
from src.data.load import load_raw_data
from src.data.validate import validate_schema
from scipy.stats import chisquare

def main():
    # Load & validate data
    df = load_raw_data()
    validate_schema(df)
    prize_cols = ["1st", "2nd", "2nd.1", "2nd.2"]
    city_results = []
    print("\n--- City Digit Counts & Chi-Square Analysis ---\n")
    for city, city_df in df.groupby("City"):
        numbers = city_df[prize_cols].values.flatten()
        numbers = numbers[~np.isnan(numbers)]
        # LAST DIGIT ANALYSIS
        last_digits = numbers % 10
        counts = np.bincount(last_digits, minlength=10)
        expected = np.full(10, counts.mean())
        chi_stat, p_value = chisquare(counts, expected)
        # TRANSITION ANALYSIS: last digit of previous draw → current draw
        transitions = np.zeros((10, 10), dtype=int)
        for i in range(1, len(last_digits)):
            prev_digit = int(last_digits[i-1])
            curr_digit = int(last_digits[i])
            transitions[prev_digit][curr_digit] += 1
        city_results.append({
            "City": city,
            "Draws": len(city_df),
            "Chi2_LastDigit": chi_stat,
            "p_LastDigit": p_value,
            "Digit_Counts": counts,
            "Transitions": transitions
        })
        # Print counts for this city
        print(f"City: {city}")
        print(f"Digit Counts: {counts}")
        print(f"Chi2: {chi_stat:.2f}, p-value: {p_value:.3f}")
        print(f"Transition Matrix (prev digit → current digit):\n{transitions}\n")
    print("\n--- Summary Table ---\n")
    summary_df = pd.DataFrame([
        {"City": r["City"], "Draws": r["Draws"], "Chi2_LastDigit": r["Chi2_LastDigit"], "p_LastDigit": r["p_LastDigit"]}
        for r in city_results
    ]).sort_values("p_LastDigit")
    print(summary_df)
if __name__ == "__main__":
    main()