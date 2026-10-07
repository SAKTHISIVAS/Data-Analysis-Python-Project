"""
data_cleaning.py
Creates a validated working copy of the PVDAQ dataset.

The original df is never modified. All preparation is performed on prep_df.
"""

from pathlib import Path
import pandas as pd

from data_loading import load_dataset


def prepare_dataset(df):
    """
    Prepare a working copy of the dataset.

    Operations are based on the EDA notebook:
    - convert timestamp to datetime
    - sort chronologically
    - calculate time differences
    - validate the 15-minute sampling interval
    - fill missing date/hour/month/year metadata from timestamp
    - keep measurement missing values unchanged
    """
    prep_df = df.copy()

    prep_df["timestamp"] = pd.to_datetime(
        prep_df["timestamp"],
        errors="coerce"
    )

    prep_df = prep_df.sort_values("timestamp").copy()

    prep_df["time_difference"] = prep_df["timestamp"].diff()

    # Reconstruct calendar metadata only where missing.
    prep_df["date"] = prep_df["date"].fillna(
        prep_df["timestamp"].dt.strftime("%Y-%m-%d")
    )
    prep_df["hour"] = prep_df["hour"].fillna(
        prep_df["timestamp"].dt.hour
    )
    prep_df["month"] = prep_df["month"].fillna(
        prep_df["timestamp"].dt.month
    )
    prep_df["year"] = prep_df["year"].fillna(
        prep_df["timestamp"].dt.year
    )

    return prep_df


def validate_dataset(df, prep_df):
    """Run the main data-quality checks used in the notebook."""
    inverter_cols = [
        col for col in prep_df.columns
        if col.startswith("inv_")
    ]

    poa_col = "poa_irradiance_o_149574"

    print("Original dataset shape:", df.shape)
    print("Prepared dataset shape:", prep_df.shape)

    print("\nDuplicate complete rows:", prep_df.duplicated().sum())
    print("Duplicate timestamps:", prep_df["timestamp"].duplicated().sum())

    print("\nTimestamp range:")
    print("Start:", prep_df["timestamp"].min())
    print("End:  ", prep_df["timestamp"].max())

    print("\nSampling interval counts:")
    print(prep_df["time_difference"].value_counts(dropna=False))

    print("\nMissing values:")
    print(prep_df.isna().sum().loc[lambda s: s > 0])

    # Check negative values in power and environmental measurements.
    measurement_cols = (
        ["total_inverter_ac_power"]
        + inverter_cols
        + [
            poa_col,
            "ambient_temperature_o_149575",
            "wind_speed_o_149576",
            "wind_direction_o_149577",
            "meter_revenue_grade_ac_output_meter_149578",
        ]
    )

    negative_counts = (prep_df[measurement_cols] < 0).sum()
    print("\nNegative-value counts:")
    print(negative_counts[negative_counts > 0])

    return {
        "inverter_cols": inverter_cols,
        "poa_col": poa_col,
    }


if __name__ == "__main__":
    df = load_dataset()
    prep_df = prepare_dataset(df)
    validate_dataset(df, prep_df)
