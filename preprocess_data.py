
# This script defines a utility function to load and preprocess crop protection data from a file. It is designed for
# use in data science and machine learning workflows where raw input data needs to be cleaned and standardized before analysis or modeling.
# Function Description
# load_and_prepare_data(file_path: str) -> pd.DataFrame
# This function:
# Loads data from a CSV (or Excel, with slight modification).
# Converts specific columns to categorical or numeric types.
# Handles missing values using simple imputation.
# Returns a clean, ready-to-use pandas.DataFrame.
# Parameters:
# file_path: A string representing the path to the data file (CSV or Excel).
# Returns:
# A preprocessed pd.DataFrame where:
# Categorical columns are correctly typed.
# Numeric and categorical missing values are filled.
# Numeric conversions are applied to relevant columns.
# Key Steps Performed
# File Reading
# Uses pd.read_csv() to load the file into a DataFrame.
# (Can be modified to support .xlsx using pd.read_excel().)
# Type Conversion
# Converts selected columns (e.g., "state", "product") to category type for memory efficiency and modeling clarity.
# Converts "year", "tenure", and similar columns to numeric, coercing errors to NaN.
# Missing Value Imputation
# Numeric columns: missing values are filled with the median of each column.
# Categorical columns: missing values are filled with the mode (most frequent value) of each column.
# Use Cases
# Preparing agricultural or crop protection datasets for regression, classification, or probabilistic modeling.
# Standardizing data ingestion in data pipelines.
# Ensuring consistent preprocessing across experiments.
# Dependencies
# pandas: for data manipulation.
# typing.Optional: used in the function signature, although not required for the current implementation.

import pandas as pd


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess crop protection data from a CSV or Excel file.

    This function reads the dataset from either CSV or Excel format, applies type conversions,
    handles missing values, and prepares the data for modeling.

    :param file_path: Path to the CSV or Excel file containing the dataset.
    :type file_path: str
    :return: A cleaned and preprocessed pandas DataFrame.
    :rtype: pd.DataFrame
    """
    df = pd.read_csv(file_path)
    # Convert categorical columns to category type
    categorical_columns = [
        "state", "climate_zone", "farmer_id", "product", "manufacturer", "indication"
    ]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Convert year columns to numeric
    for col in ["year", "first_purchase_year_product", "tenure"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing numeric values with median
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Fill missing categorical values with mode
    for col in categorical_columns:
        if col in df.columns:
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value[0])

    return df

