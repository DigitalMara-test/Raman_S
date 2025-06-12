# This script implements a hierarchical Bayesian model to estimate price elasticity of demand across different crop
# protection products using NumPyro (a probabilistic programming library based on JAX). The script is structured to
# perform the entire pipeline from data loading and preprocessing to Bayesian inference using MCMC with the NUTS sampler.
#
# 🔧 Key Functionalities
# 1. Data Loading and Preprocessing
# load_and_prepare_data(file_path: str) -> pd.DataFrame:
# Loads a dataset from a CSV file, converts categorical columns to proper types, handles missing values, and
# ensures numeric columns are ready for modeling.
#
# 2. Model Input Preparation
# prepare_model_data(df: pd.DataFrame):
# Prepares features for modeling:
# Encodes products as categorical indices.
# Standardizes the our_price and quantity variables.#
# Returns these transformed values along with the number of unique products.#
# 3. Bayesian Model Definition
# hierarchical_demand_model(...):
# Defines a hierarchical linear regression model:#
# Each product has its own intercept (alpha) and slope (beta) parameters.
# These parameters are drawn from shared normal distributions, allowing for information sharing (hierarchy).
# The observed (standardized) quantity is modeled as normally distributed around the product-specific linear prediction.
# 4. Posterior Inference with MCMC
# run_mcmc(...):
# Runs No-U-Turn Sampler (NUTS) to infer the posterior distributions of model parameters.#
# Uses init_to_median() for stable initialization.#
# Runs a single MCMC chain with 1000 warmup steps and 1000 samples.#
# Prints a summary of the inferred parameters.#
# 5. Execution Example
# In the __main__ section:#
# Loads data from "data/data.csv".#
# Prepares it.
# Runs MCMC on the hierarchical model and outputs posterior summaries.#
# Use Cases
# Estimating product-level price sensitivity.#
# Understanding demand elasticity across a product portfolio.
# Bayesian modeling of hierarchical data in the agricultural or retail sector.

import numpy as np
import pandas as pd
# import jax.numpy as jnp
# import jax.random as random
# import numpyro
# import numpyro.distributions as dist
# from numpyro.infer import MCMC, NUTS
# from numpyro.infer import init_to_median
# from typing import Any, Tuple


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess crop protection data from a CSV file.

    :param file_path: Path to the CSV file containing the dataset.
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


# def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Any, Any, Any, int]:
#     """
#     Prepare variables from the cleaned DataFrame for Bayesian modeling.
#
#     :param df: Preprocessed pandas DataFrame.
#     :type df: pd.DataFrame
#     :return: Tuple of (DataFrame, product indices, scaled prices, scaled quantities, number of products).
#     :rtype: Tuple[pd.DataFrame, Any, Any, Any, int]
#     """
#     df = df.copy()
#     df["product_code"] = df["product"].astype("category").cat.codes
#     df["price_scaled"] = (df["our_price"] - df["our_price"].mean()) / df["our_price"].std()
#     df["quantity_scaled"] = (df["quantity"] - df["quantity"].mean()) / df["quantity"].std()
#
#     product_idx = df["product_code"].values
#     price = df["price_scaled"].values
#     quantity = df["quantity_scaled"].values
#     n_products = df["product_code"].nunique()
#
#     return df, product_idx, price, quantity, n_products
#
#
# def hierarchical_demand_model(
#     product_idx: jnp.ndarray,
#     price: jnp.ndarray,
#     quantity: jnp.ndarray,
#     n_products: int
# ) -> None:
#     """
#     Hierarchical Bayesian regression model to estimate price elasticity of demand across products.
#
#     :param product_idx: Encoded product indices.
#     :param price: Scaled price values.
#     :param quantity: Scaled observed target.
#     :param n_products: Number of unique products.
#     """
#     mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0, 10))
#     mu_beta = numpyro.sample("mu_beta", dist.Normal(0, 10))
#     sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1.0))
#     sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(1.0))
#
#     alpha = numpyro.sample("alpha", dist.Normal(mu_alpha, sigma_alpha).expand([n_products]))
#     beta = numpyro.sample("beta", dist.Normal(mu_beta, sigma_beta).expand([n_products]))
#
#     mu = alpha[product_idx] + beta[product_idx] * price
#
#     sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
#     numpyro.sample("obs", dist.Normal(mu, sigma), obs=quantity)
#
#
# def run_mcmc(
#     model_fn: Any,
#     product_idx: np.ndarray,
#     price: np.ndarray,
#     quantity: np.ndarray,
#     n_products: int,
#     seed: int = 0
# ) -> MCMC:
#     """
#     Run NUTS MCMC to infer posterior distributions of the hierarchical model.
#
#     :param model_fn: NumPyro model function.
#     :param product_idx: Encoded product indices.
#     :param price: Scaled price values.
#     :param quantity: Scaled observed target.
#     :param n_products: Number of unique products.
#     :param seed: PRNG seed for reproducibility.
#     :return: Trained MCMC object.
#     """
#     rng_key = random.PRNGKey(seed)
#     kernel = NUTS(model_fn, target_accept_prob=0.95, init_strategy=init_to_median())
#     mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=1)
#     mcmc.run(
#         rng_key,
#         product_idx=product_idx,
#         price=price,
#         quantity=quantity,
#         n_products=n_products
#     )
#     mcmc.print_summary()
#     return mcmc
#
#
# # --- Example run with your real dataset ---
# if __name__ == "__main__":
#     file_path = "data/data.csv"
#     df = load_and_prepare_data(file_path)
#     df, product_idx, price, quantity, n_products = prepare_model_data(df)
#     mcmc_result = run_mcmc(hierarchical_demand_model, product_idx, price, quantity, n_products)
