# This script performs data quality inspection on a dataset by generating informative diagnostics. It is primarily intended for exploratory data analysis (EDA) before modeling, and helps identify data issues such as missing values, duplicates, inconsistent categories, and potential multicollinearity.
#  Functionality Overview
# inspect_data_quality(df: pd.DataFrame)
# This function displays comprehensive diagnostics for any given DataFrame. It helps users understand:
# Data completeness
#  Basic statistics
# Cardinality of features
# Duplicate rows
# Value distributions
# Correlation patterns among numerical variables
# Parameters:
# df: A cleaned or raw pandas.DataFrame to inspect.
# What the Script Does
# Basic Info
# Uses df.info() to print column types, null counts, and memory usage.
# Missing Values
# Lists the top 20 columns with the highest number of missing entries.
# Descriptive Statistics
# Summary statistics (count, mean, std, min, max, etc.) for numeric columns.
# Categorical Distribution
# Displays the top 5 most frequent values for each categorical column.
# Duplicate Rows
# Reports the total number of duplicated entries in the dataset.
# Cardinality Check
# Lists the number of unique values for each column, helping to flag high-cardinality variables.
# Correlation Heatmap
# Computes and visualizes the correlation matrix of the top 10 most correlated numeric features.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from start_numpyro import load_and_prepare_data


def inspect_data_quality(df: pd.DataFrame) -> None:
    """
    Display key diagnostics about data quality for a given DataFrame.

    :param df: The input pandas DataFrame to inspect.
    :type df: pd.DataFrame
    """
    print("BASIC INFO")
    print("-" * 60)
    print(df.info())
    print("\n\nMISSING VALUES (Top 20)")
    print("-" * 60)
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.head(20))

    print("\n\nNUMERIC DESCRIPTIVE STATS")
    print("-" * 60)
    print(df.describe().T)

    print("\n\nCATEGORICAL VALUE COUNTS (Top 5 each)")
    print("-" * 60)
    cat_cols = df.select_dtypes(include=["category", "object"]).columns
    for col in cat_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head(5))

    print("\n\nDUPLICATES")
    print("-" * 60)
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\n\nUNIQUE VALUES PER COLUMN")
    print("-" * 60)
    print(df.nunique().sort_values(ascending=True))

    print("\n\nCORRELATION MATRIX (heatmap of top 10)")
    print("-" * 60)
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    corr = numeric_df.corr().abs()
    top_corr = corr.unstack().sort_values(ascending=False).drop_duplicates()
    top10_features = set([i for i, j in top_corr.index[:20]])
    sns.heatmap(numeric_df[list(top10_features)].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Top correlated features")
    plt.show()

if __name__ == "__main__":
    df = load_and_prepare_data("data/data.csv")
    inspect_data_quality(df)