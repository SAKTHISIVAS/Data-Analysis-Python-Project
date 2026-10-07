"""
data_loading.py
Loads the PVDAQ solar dataset and performs basic structural inspection.
"""

from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "Dataset" / "PVDAQ_2107_2024_2025_15min_cleaned.csv"


def load_dataset(path=DATA_PATH):
    """Load the original dataset without modifying it."""
    df = pd.read_csv(path)
    return df


def inspect_dataset(df):
    """Print basic information about the dataset."""
    print("Dataset shape:", df.shape)
    print("\nColumn names:")
    for column in df.columns:
        print("-", column)

    print("\nData types and non-null counts:")
    df.info()

    print("\nMissing values by column:")
    print(df.isnull().sum())


if __name__ == "__main__":
    df = load_dataset()
    inspect_dataset(df)
