"""
data_visualization.py
Creates the main visualizations used in the EDA notebook.
"""

import matplotlib.pyplot as plt
import pandas as pd

from data_loading import load_dataset
from data_cleaning import prepare_dataset
from exploratory_analysis import (
    POA_COL,
    TOTAL_POWER_COL,
    get_inverter_columns,
    inverter_performance,
    irradiance_power_relationship,
    time_based_analysis,
    correlation_analysis,
)


def plot_inverter_performance(prep_df, inverter_cols):
    mean_power, comparison, _ = inverter_performance(
        prep_df, inverter_cols
    )

    plt.figure(figsize=(10, 8))
    mean_power.plot(kind="barh")
    plt.title("Mean Power Output by Inverter")
    plt.xlabel("Mean Power")
    plt.ylabel("Inverter")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    comparison["Difference (%)"].sort_values().plot(kind="barh")
    plt.axvline(0, linestyle="--")
    plt.title("Inverter Power Difference from Fleet Mean")
    plt.xlabel("Difference from Fleet Mean (%)")
    plt.ylabel("Inverter")
    plt.tight_layout()
    plt.show()


def plot_power_distributions(prep_df, inverter_cols):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    prep_df[TOTAL_POWER_COL].hist(bins=40, ax=axes[0])
    axes[0].set_title("Distribution of Total Inverter AC Power")
    axes[0].set_xlabel("Total Inverter AC Power")
    axes[0].set_ylabel("Frequency")

    prep_df[POA_COL].dropna().hist(bins=40, ax=axes[1])
    axes[1].set_title("Distribution of POA Irradiance")
    axes[1].set_xlabel("POA Irradiance")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 7))
    prep_df[inverter_cols].boxplot(
        rot=90,
        showfliers=False
    )
    plt.title("Power Distribution Across Individual Inverters")
    plt.xlabel("Inverter")
    plt.ylabel("Recorded Power")
    plt.tight_layout()
    plt.show()

    positive_inverter_power = prep_df[inverter_cols].where(
        prep_df[inverter_cols] > 0
    )

    plt.figure(figsize=(14, 7))
    positive_inverter_power.boxplot(
        rot=90,
        showfliers=False
    )
    plt.title("Positive Power Distribution Across Individual Inverters")
    plt.xlabel("Inverter")
    plt.ylabel("Recorded Positive Power")
    plt.tight_layout()
    plt.show()


def plot_irradiance_relationship(prep_df):
    relationship_df, pearson, spearman = (
        irradiance_power_relationship(prep_df)
    )

    plt.figure(figsize=(9, 6))
    plt.scatter(
        relationship_df[POA_COL],
        relationship_df[TOTAL_POWER_COL],
        alpha=0.4
    )
    plt.title("POA Irradiance vs Total Inverter AC Power")
    plt.xlabel("POA Irradiance")
    plt.ylabel("Total Inverter AC Power")
    plt.tight_layout()
    plt.show()

    print("Pearson correlation:", round(pearson, 4))
    print("Spearman correlation:", round(spearman, 4))

    selected_inverters = [
        "inv_16_ac_power_inv_149658",
        "inv_10_ac_power_inv_149628",
        "inv_08_ac_power_inv_149618",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, inverter in zip(axes, selected_inverters):
        plot_df = prep_df[[POA_COL, inverter]].dropna()
        ax.scatter(
            plot_df[POA_COL],
            plot_df[inverter],
            alpha=0.35
        )
        ax.set_title(inverter)
        ax.set_xlabel("POA Irradiance")
        ax.set_ylabel("Inverter Power")

    plt.tight_layout()
    plt.show()

    high_irradiance_df = prep_df.loc[
        prep_df[POA_COL] >= 200
    ]

    zero_pct_high = (
        high_irradiance_df[selected_inverters]
        .eq(0)
        .mean()
        .mul(100)
        .sort_values()
    )

    plt.figure(figsize=(9, 4.5))
    zero_pct_high.plot(kind="barh")
    plt.title(
        "Zero-Power Percentage During Higher Irradiance "
        "(POA >= 200)"
    )
    plt.xlabel("Zero-Power Readings (%)")
    plt.ylabel("Inverter")
    plt.tight_layout()
    plt.show()


def plot_time_analysis(prep_df):
    (
        hourly_profile,
        daily_summary,
        complete_days,
        daily_solar_complete,
        monthly_summary,
        monthly_inverter_summary,
    ) = time_based_analysis(prep_df)

    hourly_mean_power = (
        prep_df.groupby("hour")[TOTAL_POWER_COL]
        .mean()
        .sort_index()
    )

    plt.figure(figsize=(10, 5))
    hourly_mean_power.plot(kind="line", marker="o")
    plt.title("Average Total Inverter AC Power by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Total Inverter AC Power")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    hourly_profile["Average_POA_Irradiance"].plot(
        ax=axes[0], marker="o"
    )
    axes[0].set_title("Average POA Irradiance by Hour")
    axes[0].set_ylabel("Average POA Irradiance")
    axes[0].grid(True, alpha=0.3)

    hourly_profile["Average_Total_Power"].plot(
        ax=axes[1], marker="o"
    )
    axes[1].set_title("Average Total Inverter Power by Hour")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Average Total Power")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(13, 5))
    complete_days["Average_Total_Power"].plot(
        kind="line",
        marker="o",
        markersize=4
    )
    plt.title("Daily Average Total Inverter Power (Complete Days Only)")
    plt.xlabel("Date")
    plt.ylabel("Daily Average Total Power")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(
        daily_solar_complete.index,
        daily_solar_complete["Average_Total_Power"],
        marker="o",
        label="Average Total Inverter Power"
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Average Total Inverter Power")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        daily_solar_complete.index,
        daily_solar_complete["Average_POA_Irradiance"],
        marker="o",
        label="Average POA Irradiance"
    )
    ax2.set_ylabel("Average POA Irradiance")

    plt.title("Daily Average POA Irradiance and Total Inverter Power")
    plt.tight_layout()
    plt.show()

    monthly_inverter_summary.columns = [
        f"Month_{int(month)}_Average_Power"
        for month in monthly_inverter_summary.columns
    ]

    monthly_inverter_summary.plot(
        kind="bar",
        figsize=(16, 6),
        width=0.8
    )
    plt.title("Monthly Average Power by Inverter")
    plt.xlabel("Inverter")
    plt.ylabel("Average Power")
    plt.xticks(rotation=90)
    plt.legend(title="Month")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_missing_data(prep_df):
    missing_summary = pd.DataFrame({
        "Missing_Count": prep_df.isnull().sum(),
        "Missing_Percentage": (
            prep_df.isnull().sum() / len(prep_df) * 100
        ),
    })

    missing_summary = missing_summary[
        missing_summary["Missing_Count"] > 0
    ].sort_values("Missing_Count", ascending=False)

    plt.figure(figsize=(12, 6))
    missing_summary["Missing_Count"].sort_values().plot(
        kind="barh"
    )
    plt.title("Missing Values by Column")
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Column")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

    missing_poa_by_hour = (
        prep_df.loc[prep_df[POA_COL].isna()]
        .groupby("hour")
        .size()
        .reindex(range(24), fill_value=0)
    )

    plt.figure(figsize=(12, 4))
    missing_poa_by_hour.plot(kind="bar")
    plt.title("Missing POA Irradiance Records by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Missing Records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_correlations(prep_df, inverter_cols):
    correlation_matrix, inverter_corr = correlation_analysis(
        prep_df, inverter_cols
    )

    plt.figure(figsize=(9, 7))
    image = plt.imshow(
        correlation_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )
    plt.colorbar(image, label="Pearson Correlation")

    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=45,
        ha="right"
    )
    plt.yticks(
        range(len(correlation_matrix.index)),
        correlation_matrix.index
    )

    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(
                j,
                i,
                f"{correlation_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    plt.title("Correlation Matrix of Key Solar and Environmental Variables")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    inverter_corr.plot(kind="barh")
    plt.title("POA Irradiance Correlation by Inverter")
    plt.xlabel("Pearson Correlation")
    plt.ylabel("Inverter")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_dataset()
    prep_df = prepare_dataset(df)
    inverter_cols = get_inverter_columns(prep_df)

    plot_inverter_performance(prep_df, inverter_cols)
    plot_power_distributions(prep_df, inverter_cols)
    plot_irradiance_relationship(prep_df)
    plot_time_analysis(prep_df)
    plot_missing_data(prep_df)
    plot_correlations(prep_df, inverter_cols)
