"""
exploratory_analysis.py
Performs the numerical and statistical analysis from the EDA notebook.

No machine-learning model is included here.
"""

import pandas as pd

from data_loading import load_dataset
from data_cleaning import prepare_dataset


POA_COL = "poa_irradiance_o_149574"
TEMP_COL = "ambient_temperature_o_149575"
WIND_SPEED_COL = "wind_speed_o_149576"
WIND_DIRECTION_COL = "wind_direction_o_149577"
TOTAL_POWER_COL = "total_inverter_ac_power"


def get_inverter_columns(df):
    return [col for col in df.columns if col.startswith("inv_")]


def descriptive_statistics(prep_df, inverter_cols):
    measurement_cols = (
        [TOTAL_POWER_COL]
        + inverter_cols
        + [
            POA_COL,
            TEMP_COL,
            WIND_SPEED_COL,
            WIND_DIRECTION_COL,
            "meter_revenue_grade_ac_output_meter_149578",
        ]
    )

    return prep_df[measurement_cols].describe().T


def inverter_performance(prep_df, inverter_cols):
    mean_power = (
        prep_df[inverter_cols]
        .mean()
        .sort_values(ascending=True)
    )

    solar_active_mask = prep_df[POA_COL] >= 200
    solar_active_df = prep_df.loc[solar_active_mask].copy()

    high_irradiance_mean = (
        solar_active_df[inverter_cols]
        .mean()
        .sort_values(ascending=True)
    )

    fleet_mean = high_irradiance_mean.mean()

    comparison = pd.DataFrame({
        "Mean Power": high_irradiance_mean
    })
    comparison["Difference from Fleet Mean"] = (
        comparison["Mean Power"] - fleet_mean
    )
    comparison["Difference (%)"] = (
        comparison["Difference from Fleet Mean"]
        / fleet_mean
        * 100
    )

    zero_high_irradiance = (
        solar_active_df[inverter_cols]
        .eq(0)
        .sum()
        .to_frame("Zero_Count")
    )
    zero_high_irradiance["Zero_Percentage"] = (
        zero_high_irradiance["Zero_Count"]
        / len(solar_active_df)
        * 100
    )

    return mean_power, comparison, zero_high_irradiance


def power_summary(prep_df):
    total_power = prep_df[TOTAL_POWER_COL]

    return pd.DataFrame({
        "Count": [
            total_power.eq(0).sum(),
            total_power.gt(0).sum(),
            total_power.lt(0).sum(),
        ]
    }, index=[
        "Zero Power",
        "Positive Power",
        "Negative Power",
    ])


def irradiance_power_relationship(prep_df):
    relationship_df = prep_df[
        [POA_COL, TOTAL_POWER_COL]
    ].dropna()

    pearson = relationship_df[POA_COL].corr(
        relationship_df[TOTAL_POWER_COL],
        method="pearson"
    )
    spearman = relationship_df[POA_COL].corr(
        relationship_df[TOTAL_POWER_COL],
        method="spearman"
    )

    return relationship_df, pearson, spearman


def time_based_analysis(prep_df):
    hourly_profile = prep_df.groupby("hour").agg(
        Average_POA_Irradiance=(POA_COL, "mean"),
        Average_Total_Power=(TOTAL_POWER_COL, "mean"),
    )

    daily_summary = prep_df.groupby("date").agg(
        Average_Total_Power=(TOTAL_POWER_COL, "mean"),
        Maximum_Total_Power=(TOTAL_POWER_COL, "max"),
        Total_Observations=(TOTAL_POWER_COL, "count"),
    )

    complete_days = daily_summary[
        daily_summary["Total_Observations"] == 96
    ].copy()

    daily_solar = prep_df.groupby("date").agg(
        Average_POA_Irradiance=(POA_COL, "mean"),
        Average_Total_Power=(TOTAL_POWER_COL, "mean"),
        Total_Observations=(TOTAL_POWER_COL, "count"),
    )

    daily_solar_complete = daily_solar[
        daily_solar["Total_Observations"] == 96
    ].dropna(
        subset=["Average_POA_Irradiance", "Average_Total_Power"]
    )

    monthly_summary = prep_df.groupby("month").agg(
        Average_POA_Irradiance=(POA_COL, "mean"),
        Average_Total_Power=(TOTAL_POWER_COL, "mean"),
        Total_Observations=(TOTAL_POWER_COL, "count"),
        Unique_Days=("date", "nunique"),
    )

    monthly_inverter_summary = (
        prep_df.groupby("month")[get_inverter_columns(prep_df)]
        .mean()
        .T
    )

    return (
        hourly_profile,
        daily_summary,
        complete_days,
        daily_solar_complete,
        monthly_summary,
        monthly_inverter_summary,
    )


def missing_data_analysis(prep_df, inverter_cols):
    missing_summary = pd.DataFrame({
        "Missing_Count": prep_df.isnull().sum(),
        "Missing_Percentage": (
            prep_df.isnull().sum() / len(prep_df) * 100
        ),
    })

    missing_summary = missing_summary[
        missing_summary["Missing_Count"] > 0
    ].sort_values("Missing_Count", ascending=False)

    missing_poa_mask = prep_df[POA_COL].isna()

    total_cells = prep_df.size
    missing_cells = int(prep_df.isna().sum().sum())
    non_missing_cells = total_cells - missing_cells

    completeness = {
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "non_missing_cells": non_missing_cells,
        "completeness_percent": non_missing_cells / total_cells * 100,
        "complete_rows": int(prep_df.notna().all(axis=1).sum()),
        "rows_with_missing": int(prep_df.isna().any(axis=1).sum()),
        "missing_poa_records": int(missing_poa_mask.sum()),
        "missing_poa_zero_power": int(
            prep_df.loc[missing_poa_mask, TOTAL_POWER_COL].eq(0).sum()
        ),
    }

    return missing_summary, completeness


def correlation_analysis(prep_df, inverter_cols):
    key_columns = [
        TOTAL_POWER_COL,
        POA_COL,
        TEMP_COL,
        WIND_SPEED_COL,
        WIND_DIRECTION_COL,
    ]

    correlation_matrix = prep_df[key_columns].corr(method="pearson")

    inverter_irradiance_corr = (
        prep_df[inverter_cols + [POA_COL]]
        .corr(method="pearson")[POA_COL]
        .drop(labels=[POA_COL])
        .sort_values()
    )

    return correlation_matrix, inverter_irradiance_corr


if __name__ == "__main__":
    df = load_dataset()
    prep_df = prepare_dataset(df)
    inverter_cols = get_inverter_columns(prep_df)

    print("\n=== Descriptive Statistics ===")
    print(descriptive_statistics(prep_df, inverter_cols).round(2))

    print("\n=== Power Summary ===")
    print(power_summary(prep_df))

    print("\n=== Inverter Performance ===")
    mean_power, comparison, zero_high = inverter_performance(
        prep_df, inverter_cols
    )
    print("\nMean inverter power:")
    print(mean_power.round(2))
    print("\nHigh-irradiance comparison:")
    print(comparison.round(2))
    print("\nHigh-irradiance zero readings:")
    print(zero_high.round(2))

    print("\n=== POA vs Total Power ===")
    _, pearson, spearman = irradiance_power_relationship(prep_df)
    print("Pearson correlation:", round(pearson, 4))
    print("Spearman correlation:", round(spearman, 4))

    print("\n=== Missing Data ===")
    missing_summary, completeness = missing_data_analysis(
        prep_df, inverter_cols
    )
    print(missing_summary.round(2))
    print("\nCompleteness:", round(
        completeness["completeness_percent"], 2
    ), "%")

    print("\n=== Correlation Matrix ===")
    correlation_matrix, inverter_corr = correlation_analysis(
        prep_df, inverter_cols
    )
    print(correlation_matrix.round(3))
    print("\nInverter vs POA correlation:")
    print(inverter_corr.round(3))
