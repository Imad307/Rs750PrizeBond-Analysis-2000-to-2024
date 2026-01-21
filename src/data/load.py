import pandas as pd
from pathlib import Path


# Get project root directory (PrizeBond750Analysis)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "prize_bond_750.csv"


def load_raw_data() -> pd.DataFrame:
    """
    Loads raw prize bond dataset without any modification.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    return df
