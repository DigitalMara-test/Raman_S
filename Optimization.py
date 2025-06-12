# Script Description: Evolutionary Optimization for Profit Maximization
# This Python script implements an evolutionary algorithm to optimize business objectives — such as profit,
# quantity, or churn — using pre-trained machine learning models (XGBoost) for quantity prediction and churn prediction.
#  It integrates constraint handling, variable freezing, automatic bound inference, and returns optimal input parameters for maximizing a selected business metric.
# Core Function: evolutionary_optimizer(...)
# Purpose:
# To find the best combination of input variables that maximizes a chosen metric (e.g., profit) while satisfying user-defined constraints and respecting frozen parameters.
# Parameters:
# df_new: The dataset for prediction and evaluation.
# model_dir: Directory where pre-trained models (model_quantity.pkl, model_churn.pkl) are stored.
# metric: The target metric to optimize, such as "profit", "quantity", or "churn".
# constraints: A dictionary specifying conditions on output or input variables, e.g., "churn": ("<", 0.3).
# frozen_params: Fixed values for some input features (e.g., "year": 2024) that should not be optimized.
# variable_bounds: Ranges for all optimizable input variables, e.g., {"our_price": (10, 30)}.
# population_size, generations, cxpb, mutpb, seed: DEAP algorithm settings for the evolutionary search.
# Key Components:
# 1. Model Loading
# Loads XGBoost models for quantity and churn prediction from disk.
# 2. Output Computation Function
# # compute_outputs(...)
# Calculates:
# profit: Adjusted for churn probability.
# quantity: Inverse of log-transformed demand predictions.
# churn: Average churn probability.
# revenue, margin, and any extra variables.
# 3. Constraint Checking
# # constraint_ok(results)`
# Evaluates whether predicted results satisfy all provided constraints. Supports bounds with suffixes:
# _lower (≥ or >)
# _upper (≤ or <)
# as well as equality (==).
# 4. Fitness Evaluation
# Each individual in the population is scored based on the target metric if constraints are satisfied. Otherwise, it receives a penalty.
# 5. Evolutionary Algorithm (via DEAP)
# Includes:
# Random initialization,
# Tournament selection,
# Gaussian mutation,
# Blend crossover (cxBlend),
# Logging of stats and best individuals across generations.
# 6. Best Result Extraction
# Returns:
# Best parameter values (optimized and frozen),
# The optimized metric value,
# All computed output variables,
# Evolution logs for analysis.
# Output Format
# Returns a dictionary:
# {
#     "metric": "profit",
#     "best_value": ...,
#     "outputs": {
#         "profit": ...,
#         "churn": ...,
#         "quantity": ...,
#         ...
#     },
#     "param1": val1,
#     "param2": val2,
#     ...
#     "logbook": <evolution logs>
# }
# Execution Block (__main__)
# Loads dataset from "data/data.csv".
# Samples 10,000 rows for optimization.
# Infers bounds for all numeric variables.
# Removes non-optimizable outputs (churn, profit, quantity) from bounds.
# Runs the optimizer with:
# Metric: "profit"
# Constraints on outputs and variable bounds
# Frozen parameters (e.g., year and churn probability)
# Prints best result.
# Dependencies
# pandas, numpy
# joblib (for loading models)
# DEAP (evolutionary optimization framework)
# XGBoost (via pre-trained models)
# Use Case
# Ideal for marketing, pricing, and demand modeling teams who:
# Have predictive models for churn and demand,
# Want to optimize pricing, spend, or operational variables,
# Need to enforce real-world business rules or constraints during optimization.

import pandas as pd
import numpy as np

from typing import Optional, Dict, Tuple

import joblib
import os
import random
from deap import base, creator, tools, algorithms

def evolutionary_optimizer(
    df_new: pd.DataFrame,
    model_dir: str = "models",
    metric: str = "profit",
    constraints: Optional[Dict[str, Tuple[str, float]]] = None,
    frozen_params: Optional[Dict[str, float]] = None,
    variable_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    population_size: int = 90,
    generations: int = 30,
    cxpb: float = 0.5,
    mutpb: float = 0.2,
    seed: int = 42
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    model_q = joblib.load(os.path.join(model_dir, "model_quantity.pkl"))
    model_c = joblib.load(os.path.join(model_dir, "model_churn.pkl"))

    df_opt = df_new.copy()
    opt_features = [key for key in variable_bounds.keys() if frozen_params is None or key not in frozen_params]

    bounds = {
        col: variable_bounds[col] for col in opt_features
    }

    def compute_outputs(df, **params):
        df = df.copy()
        for key, value in params.items():
            df[key] = value
        X = df[model_q.named_steps["preprocessor"].feature_names_in_]

        demand = np.expm1(model_q.predict(X))
        churn_prob = model_c.predict_proba(X)[:, 1]
        margin = (df["our_price"] - df["production_cost"]) / df["our_price"]
        revenue = df["our_price"] * demand
        profit = (df["our_price"] - df["production_cost"]) * demand * (1 - churn_prob)

        return {
            "profit": profit.sum(),
            "quantity": demand.sum(),
            "churn": churn_prob.mean(),
            "revenue": revenue.sum(),
            "margin": margin.mean(),
            **params
        }

    def constraint_ok(results):
        if not constraints:
            return True
        for key, (op, val) in constraints.items():
            var_name = key.replace("_lower", "").replace("_upper", "")
            actual = results.get(var_name)
            if actual is None:
                return False
            if key.endswith("_lower"):
                if op == ">" and not actual > val:
                    return False
                if op == ">=" and not actual >= val:
                    return False
            elif key.endswith("_upper"):
                if op == "<" and not actual < val:
                    return False
                if op == "<=" and not actual <= val:
                    return False
            else:
                if op == "<" and not actual < val:
                    return False
                if op == ">" and not actual > val:
                    return False
                if op == "<=" and not actual <= val:
                    return False
                if op == ">=" and not actual >= val:
                    return False
                if op == "==" and not np.isclose(actual, val):
                    return False
        return True

    def eval_fn(ind):
        param_dict = {k: frozen_params[k] if frozen_params and k in frozen_params else v for k, v in zip(bounds.keys(), ind)}
        results = compute_outputs(df_opt, **param_dict)
        if not constraint_ok(results):
            return (-1e10,)
        return (results[metric],)

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    for key, (low, high) in bounds.items():
        toolbox.register(f"attr_{key}", random.uniform, low, high)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [toolbox.__getattribute__(f"attr_{key}") for key in bounds], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_fn)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=1.0)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, stats=stats, verbose=True)
    best_ind = tools.selBest(pop, 1)[0]

    best_param_dict = {k: frozen_params[k] if frozen_params and k in frozen_params else v for k, v in zip(bounds.keys(), best_ind)}
    best_results = compute_outputs(df_opt, **best_param_dict)

    return {
        **best_param_dict,
        "metric": metric,
        "best_value": best_results[metric],
        "outputs": best_results,
        "logbook": log
    }

if __name__ == "__main__":
    df = pd.read_csv("data/data.csv")
    df_sample = df.sample(10000, random_state=42)

    variable_bounds = {}
    numerical_cols = df_sample.select_dtypes(include="number").columns.tolist()

    for col in numerical_cols:
        min_val = df_sample[col].min()
        max_val = df_sample[col].max()
        if pd.notnull(min_val) and pd.notnull(max_val) and min_val != max_val:
            variable_bounds[col] = (min_val, max_val)

    print(f" Variable limits {variable_bounds}")
    for frozen_key in ["churn", "quantity", "profit"]:
        variable_bounds.pop(frozen_key, None)

    result = evolutionary_optimizer(
        df_new=df_sample,
        metric="profit",
        constraints={
            "churn": ("<", 0.3),
            "quantity": ("<", 3071400),
            "our_price_lower": (">=", 12),
            "our_price_upper": ("<=", 25),
            "commodity_price_lower": (">=", 21),
            "commodity_price_upper": ("<=", 170),
            "production_cost_lower": (">=", 8),
            "production_cost_upper": ("<=", 15),
            "intensity_lower": (">=", 0.1),
            "intensity_upper": ("<=", 3),
            "marketing_spend_lower": (">=", 119),
            "marketing_spend_upper": ("<=", 58684),
        },
        frozen_params={"year": 2024,
                       'first_purchase_year_product': 2024,
                       'first_overall_purchase_year': 2024,
                        'churn_prob': 0.01},
        variable_bounds=variable_bounds
    )

    print("\nBest Results:")
    print(result)
