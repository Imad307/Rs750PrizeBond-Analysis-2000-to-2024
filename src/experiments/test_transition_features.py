from src.data.load import load_raw_data
from src.features.transition_features import add_transition_features


def main():
    df = load_raw_data()
    df = add_transition_features(df, column="1st")

    print(df[[
        "Draw No.",
        "1st",
        "last_digit",
        "prev_last_digit",
        "transition_prob",
        "transition_surprise"
    ]].head(15))


if __name__ == "__main__":
    main()
