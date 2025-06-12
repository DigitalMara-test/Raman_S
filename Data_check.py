# This script provides a diagnostic tool to explore the relationship between price (our_price) and
# demand (quantity) across multiple geographic regions (state). It performs statistical summaries
# and visual diagnostics to better understand how price affects quantity in various states.
# 🔧 Functionality Overview
# ✅ diagnose_price_quantity_relationship(...)
# This function:
# Samples data from the top N regions by volume.
# Computes basic statistics and correlation.
# Visualizes distributions and price-to-quantity relationships using plots.
# Parameters:
# df: A preprocessed pandas.DataFrame containing at least "our_price", "quantity", and "state" columns.
# num_regions: Number of top states (by record count) to include in the analysis (default = 6).
# sample_size: Number of samples to draw from each selected region (default = 3000).
# random_state: Seed for reproducibility (default = 42).
# What the Script Does
# Region Selection & Sampling
# Selects top num_regions based on data availability.
# Draws up to sample_size records from each region.
# Summary Statistics
# Displays descriptive statistics for our_price and quantity.
# Shows standard deviation of price across regions.
# Distribution Plots
# Histograms of price and quantity.
# Correlation
# Computes the Pearson correlation between price and quantity across the sampled data.
# Scatterplots per Region
# Linearly fits and visualizes the relationship between price and quantity for each region.
# Uses sns.lmplot with color-coding by region.
# Log-Log Transformation
# Applies a logarithmic transformation (log1p) to both price and quantity.
# Repeats the regional linear fits to evaluate elasticity in log space (which is often linear in economics).#
# Use Cases
# Price sensitivity diagnostics across markets.
# Pre-modeling analysis for demand forecasting.
# Exploratory data analysis for elasticity estimation.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def diagnose_price_quantity_relationship(
    df: pd.DataFrame,
    num_regions: int = 6,
    sample_size: int = 3000,
    random_state: int = 42
) -> None:
    """
    Run diagnostics to explore the price-to-demand relationship across regions.

    :param df: Preprocessed DataFrame with at least 'our_price', 'quantity', and 'state' columns.
    :param num_regions: Number of top regions (by count) to include in diagnostics.
    :param sample_size: Number of observations to sample for each region.
    :param random_state: Random seed for reproducibility.
    :return: None
    """
    df = df.copy()

    # Select top N regions and sample K observations from each
    top_regions = df["state"].value_counts().head(num_regions).index
    df_subset = (
        df[df["state"].isin(top_regions)]
        .groupby("state")
        .apply(lambda x: x.sample(min(len(x), sample_size), random_state=random_state))
        .reset_index(drop=True)
    )

    print("\n--- BASIC STATS ---")
    print("Overall price stats:")
    print(df_subset["our_price"].describe())
    print("\nOverall quantity stats:")
    print(df_subset["quantity"].describe())

    print("\n--- PRICE VARIABILITY BY REGION ---")
    price_var = df_subset.groupby("state")["our_price"].std().sort_values()
    print(price_var)

    print("\n--- DISTRIBUTIONS ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df_subset["our_price"], bins=50, kde=True, ax=axes[0])
    axes[0].set_title("Distribution of Our Price")

    sns.histplot(df_subset["quantity"], bins=50, kde=True, ax=axes[1])
    axes[1].set_title("Distribution of Quantity")
    plt.tight_layout()
    plt.show()

    print("\n--- CORRELATION ---")
    print(df_subset[["our_price", "quantity"]].corr())

    print("\n--- SCATTERPLOT: PRICE vs QUANTITY by REGION ---")
    sns.lmplot(
        data=df_subset,
        x="our_price", y="quantity", hue="state", col="state",
        col_wrap=3, height=4, scatter_kws={"s": 10}, line_kws={"color": "red"}
    )
    plt.subplots_adjust(top=0.9)
    plt.suptitle("Price vs Quantity (Linear Fit per Region)")
    plt.show()

    print("\n--- LOG PRICE vs LOG QUANTITY CHECK ---")
    df_subset["log_price"] = np.log1p(df_subset["our_price"])
    df_subset["log_quantity"] = np.log1p(df_subset["quantity"])

    sns.lmplot(
        data=df_subset,
        x="log_price", y="log_quantity", hue="state", col="state",
        col_wrap=3, height=4, scatter_kws={"s": 10}, line_kws={"color": "green"}
    )
    plt.subplots_adjust(top=0.9)
    plt.suptitle("Log(Price) vs Log(Quantity) (Linear Fit per Region)")
    plt.show()


# Example usage:
if __name__ == "__main__":
    df = pd.read_csv("data/data.csv")
    diagnose_price_quantity_relationship(df, num_regions=6, sample_size=500)
