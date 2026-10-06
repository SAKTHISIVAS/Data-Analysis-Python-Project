# AI-Based Solar Power Performance and Degradation Intelligence

## Project Overview

This project applies Python-based exploratory data analysis (EDA) to
photovoltaic (PV) system performance data from the **NREL PVDAQ
(Photovoltaic Data Acquisition)** dataset.

The analysis focuses on understanding solar power generation patterns,
data quality, environmental relationships, inverter-level performance,
and potential performance anomalies as a foundation for further solar
power performance and degradation intelligence.

## Industry

**Renewable Energy --- Solar Photovoltaic (PV) Power Generation**

## Problem Statement

Solar PV plants generate large volumes of time-series data from
inverters, power meters, irradiance sensors, and environmental sensors.
Understanding this data is important for assessing power-generation
behavior and identifying unusual differences in inverter performance.

This project analyzes PV system data to understand generation patterns,
evaluate data quality, examine relationships between irradiance and
power output, compare inverter-level performance, and identify
observations that may require further investigation.

The analysis is exploratory. The identified inverter differences are not
treated as confirmed equipment faults or proof of long-term degradation.

## Proposed Solution / Analysis Questions

Python-based exploratory data analysis is used to examine the PV system
data before predictive-model development.

-   Is the dataset complete and consistent?
-   Are there duplicate records or timestamps?
-   What are the missing-data patterns?
-   How does solar irradiance relate to total inverter power?
-   How does power generation vary by time of day and date?
-   How do individual inverters compare with the fleet under
    high-irradiance conditions?
-   Which inverter-level observations may require further investigation?
-   What relationships exist between power output and the available
    environmental variables?

## Dataset

### Dataset Name

**NREL PVDAQ --- Photovoltaic Data Acquisition Public Datasets**

### Dataset Provider

**National Renewable Energy Laboratory (NREL)**

PVDAQ is a large-scale time-series database containing PV system
metadata and performance data from experimental and commercial public PV
sites. It can include electrical performance and environmental
measurements such as irradiance, temperature, and wind data.

### Dataset Used in This Project

The project uses data associated with **PVDAQ system 2107**, including
electrical, irradiance, environmental, and meter measurements.

-   **3,461 records**
-   **35 original columns**
-   **36 columns after EDA preparation**
-   **24 inverter power measurements**
-   **15-minute sampling interval**
-   Timestamp coverage: **1 February 2024 to 8 March 2024**

### Dataset Source

The dataset was obtained from the **NREL PVDAQ public dataset
resources**.

Official resource:
https://catalog.data.gov/dataset/photovoltaic-data-acquisition-pvdaq-public-datasets

## Tools & Technologies

-   Python
-   Jupyter Notebook
-   NumPy
-   Pandas
-   Matplotlib
-   Scikit-learn --- used for the subsequent baseline machine-learning
    stage

## Project Workflow

**Industry Selection → Problem Identification → Dataset Collection →
Data Cleaning → Data Transformation → Data Analysis → Data Visualization
→ Insights → Recommendations**

## Data Cleaning & Preparation

-   Dataset structure inspection
-   Timestamp conversion and chronological sorting
-   Duplicate-row checking
-   Duplicate-timestamp checking
-   Sampling-interval verification
-   Missing-value assessment
-   Date, hour, month, and year metadata preparation
-   Negative-value checks for power and irradiance measurements
-   Creation of derived time-difference information
-   Preservation of the original dataset while using a separate prepared
    dataset

### Data Quality Results

-   Duplicate rows: **0**
-   Duplicate timestamps: **0**
-   Missing cells in prepared dataset: **101**
-   Overall cell-level completeness: **99.92%**
-   Sampling interval: **15 minutes**

Missing values were retained during EDA rather than being blindly
imputed or deleted.

## Data Analysis & Visualization

Only analyses actually performed in the project are listed below.

### 1. Distribution Analysis

Total inverter AC power was examined using a histogram and
positive-power descriptive statistics. The power distribution was
strongly right-skewed, with many zero or low-power observations.

### 2. Environmental Analysis

POA irradiance, ambient temperature, wind speed, and wind direction were
examined using descriptive statistics.

### 3. Relationship Analysis

The relationship between POA irradiance and total inverter AC power was
analyzed using a scatter plot, Pearson correlation, and Spearman
correlation.

Pearson correlation: **0.9933**. Spearman correlation: **0.9955**.

### 4. Time-Based Analysis

Power generation was analyzed by hour of day, date, complete-day
summaries, and month. Hourly analysis showed power increasing during the
morning, reaching its highest average around midday, and decreasing
during the afternoon.

### 5. Daily Analysis

Daily average power, daily maximum power, and daily average POA
irradiance were examined. For the 36 complete days, daily average POA
irradiance and daily average power had a Pearson correlation of
**0.9975**.

### 6. Inverter-Level Comparison

The 24 inverter power measurements were compared using inverter
boxplots, high-irradiance average power, difference from fleet average,
zero-power observations under high irradiance, and inverter-to-POA
correlations.

Under POA irradiance of at least 200, **INV-16**, **INV-10**, and
**INV-08** had average power below the fleet average.

### 7. Correlation Analysis

A correlation matrix was created for total inverter AC power, POA
irradiance, ambient temperature, wind speed, and wind direction.

### 8. Anomaly Exploration

High-irradiance inverter performance was used to identify inverter-level
differences that may require further investigation. These observations
are treated as anomaly candidates rather than confirmed faults or
degradation.

## Key Insights

1.  The prepared dataset contains **3,461 records** and has **99.92%
    overall cell-level completeness**.
2.  The dataset contains **no duplicate rows or duplicate timestamps**.
3.  Measurements follow a consistent **15-minute sampling interval**.
4.  POA irradiance has a very strong positive relationship with total
    inverter AC power, with Pearson correlation **0.9933**.
5.  Power generation generally increases during the morning, peaks
    around midday, and decreases during the afternoon.
6.  Across the 36 complete days, daily average POA irradiance and daily
    average power have a Pearson correlation of **0.9975**.
7.  Under high-irradiance conditions, **INV-16** showed the largest
    negative difference from the fleet average.
8.  **INV-10** and **INV-08** also showed below-fleet average power
    under the selected high-irradiance condition.
9.  These inverter differences identify candidates for further
    investigation, but do not independently prove equipment faults or
    long-term degradation.
10. The available period is relatively short, so long-term degradation
    cannot be established from this dataset alone.

## Recommendations

-   Investigate INV-16, INV-10, and INV-08 further using additional
    operational and equipment information.
-   Use longer-term PV performance data before making conclusions about
    degradation.
-   Continue monitoring inverter performance under comparable irradiance
    conditions.
-   Treat missing sensor values carefully rather than applying
    indiscriminate imputation.
-   Use the EDA findings as the foundation for subsequent
    machine-learning and anomaly-detection stages.

## Visualization Screenshots

The following section is reserved for screenshots of the actual
visualizations generated in the project.

> Add the exact filenames of your saved screenshots after uploading them
> to the `Visualizations/` folder. No visualization filenames are
> invented here.

### Visualization 1 --- Total Power Distribution

`![Visualization](Visualizations/<actual-filename>.png)`

### Visualization 2 --- POA Irradiance vs Total Power

`![Visualization](Visualizations/<actual-filename>.png)`

### Visualization 3 --- Hourly Average Power

`![Visualization](Visualizations/<actual-filename>.png)`

### Visualization 4 --- Daily Power Analysis

`![Visualization](Visualizations/<actual-filename>.png)`

### Visualization 5 --- Inverter Performance Comparison

`![Visualization](Visualizations/<actual-filename>.png)`

### Visualization 6 --- Inverter Difference from Fleet Average

`![Visualization](Visualizations/<actual-filename>.png)`

### Visualization 7 --- Correlation Matrix

`![Visualization](Visualizations/<actual-filename>.png)`

## Project Folder Structure

Files whose exact filenames were not established are intentionally left
as placeholders rather than invented.

``` text
AI-Based-Solar-Power-Performance-and-Degradation-Intelligence/
│
├── README.md
│
├── Dataset/
│   ├── PVDAQ_2107_2024_2025_15min_cleaned.csv
│   └── [other actual dataset files]
│
├── Notebook/
│   └── Data_Analysis_EDA.ipynb
│
├── Python/
│   └── [actual Python scripts, if used]
│
├── Visualizations/
│   └── [actual visualization screenshots]
│
└── Documentation/
    └── Project_Report.pdf
```

Replace bracketed entries with the exact files actually uploaded to
GitHub. Do not invent files solely to match this README.

## Author

-   **Name:** SIVA KUMAR S
-   **Student ID:** AF05320553
-   **Organization:** Anudip Foundation
-   **Course:** AIML
-   **Batch Code:** ANP-D7444M

## Project Status

**EDA Sprint 1 --- Completed**

The exploratory analysis established dataset quality, solar-generation
patterns, environmental relationships, inverter-level performance
differences, and anomaly candidates. Machine-learning model development
is treated as a subsequent stage of the project.
