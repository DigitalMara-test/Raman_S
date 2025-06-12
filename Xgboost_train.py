#  Script Description: Price–Demand Diagnostics, XGBoost Modeling, and Profit Prediction
# This Python script provides a comprehensive workflow for:
# Diagnosing price-to-demand relationships across geographic regions,
# Training predictive models for quantity demanded and churn using XGBoost,
# Evaluating model performance using partial dependence plots and SHAP values,
# Making profit predictions based on model outputs and user-specified prices.
# 1. diagnose_price_quantity_relationship(...)
# Purpose:
# This function performs exploratory diagnostics and model training to understand the relationship between price,
# quantity, and churn in different regions (e.g., U.S. states).
# Key Steps:
# Region Sampling: Selects top num_regions based on record count and samples sample_size records per region.
# Descriptive Analysis:
# Displays distributions and variability for our_price and quantity.
# Generates histograms, correlation matrices, and scatterplots.
# Model Training:
# Trains two separate models using XGBRegressor (for quantity) and XGBClassifier (for churn).
# Applies preprocessing pipelines using StandardScaler for numeric features and OneHotEncoder for categorical ones.
# Removes multicollinear features using VIF (Variance Inflation Factor).
# Evaluation:
# Computes accuracy, AUC (for churn), and R² (for quantity) on test data and via cross-validation.
# Model Export:
# Saves the trained models as model_quantity.pkl and model_churn.pkl in the models directory.
# Interpretability:
# Visualizes Partial Dependence Plots (PDPs) for top features influencing predictions.
# Computes and plots SHAP values for model explainability.
# 2. plot_top_pdp(...)
# Helper function for visualizing the most important raw numeric features based on feature importances from XGBoost,
# via PDPs. Supports better understanding of how each key feature influences predictions (e.g., demand).
# 3. predict_profit(...)
# Purpose:
# Estimates the expected total profit for a given DataFrame under a specified our_price value using the
# trained quantity and churn models.
# Calculation:
# Profit is computed as:
# profit = (our_price - production_cost) * expected_demand * (1 - churn_probability)
# The function uses:
# Quantity predictions from model_quantity.pkl
# Churn predictions from model_churn.pkl
# Dependencies
# ML libraries: xgboost, scikit-learn, shap, statsmodels
# Data manipulation: pandas, numpy
# Visualization: matplotlib, seaborn
# Joblib: for saving and loading trained models
# Example Usage
# When executed directly (if __name__ == "__main__"), the script loads data/data.csv, runs the diagnostics
# and training on 15 top regions using a sample size of 20,000 per region.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
import os

def diagnose_price_quantity_relationship(
    df: pd.DataFrame,
    num_regions: int = 1,
    sample_size: int = 500,
    random_state: int = 42,
    save_dir: str = "models"
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    df = df.copy()
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

    print("\n--- SCATTERPLOT: PRICE vs QUANTITY ---")
    sns.scatterplot(data=df_subset, x="our_price", y="quantity", hue="state")
    plt.title("Price vs Quantity")
    plt.show()

    print("\n--- XGBOOST MODEL TRAINING ---")
    df_model = df_subset.dropna(subset=["quantity", "churn"])

    numeric_features_all = df_model.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_features = df_model.select_dtypes(include=["category", "object"]).columns.tolist()

    def filter_features_for_target(features, target):
        if target == "churn":
            return [f for f in features if not any(sub in f.lower() for sub in ["churn", "quantity", "revenue", "gross_margin"])]
        elif target == "quantity":
            return [f for f in features if not any(sub in f.lower() for sub in ["churn", "quantity"])]
        return features

    def train_and_save_model(y, name):
        numeric_features = filter_features_for_target(numeric_features_all, name)

        print(f"\n--- MULTICOLLINEARITY CHECK (VIF) for {name} ---")
        X_vif = df_model[numeric_features].dropna()
        X_vif_scaled = StandardScaler().fit_transform(X_vif)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(X_vif_scaled, i) for i in range(X_vif_scaled.shape[1])]
        print(vif_data.sort_values(by="VIF", ascending=False))

        high_vif_features = vif_data[vif_data["VIF"] > 100]["feature"].tolist()
        numeric_filtered = [f for f in numeric_features if f not in high_vif_features]
        print("\nDropped features due to high VIF:", high_vif_features)

        all_features = numeric_filtered + categorical_features
        X = df_model[all_features]
        y_target = y

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numeric_filtered),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ])

        if name == "churn":
            model = Pipeline([
                ("preprocessor", preprocessor),
                ("xgb", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                       random_state=random_state, scale_pos_weight=1))  # updated from 'balanced'
            ])
        else:
            model = Pipeline([
                ("preprocessor", preprocessor),
                ("xgb", XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state))
            ])

        X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=random_state)
        model.fit(X_train, y_train)

        if name == "churn":
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            print(f"\n{name} Accuracy:", acc)
            print(f"{name} ROC AUC:", auc)

            print(f"\n--- CROSS-VALIDATION for {name} ---")
            scores = cross_val_score(model, X, y_target, cv=5, scoring="roc_auc")
            print("Cross-validated ROC AUC scores:", scores)
            print("Mean CV ROC AUC:", scores.mean())
        else:
            print(f"\n{name} Train R^2:", model.score(X_train, y_train))
            print(f"{name} Test R^2:", model.score(X_test, y_test))

            print(f"\n--- CROSS-VALIDATION for {name} ---")
            scores = cross_val_score(model, X, y_target, cv=5, scoring="r2")
            print("Cross-validated R^2 scores:", scores)
            print("Mean CV R^2:", scores.mean())

        joblib.dump(model, os.path.join(save_dir, f"model_{name}.pkl"))
        return model, X_test, preprocessor, model.named_steps["preprocessor"].get_feature_names_out(), numeric_filtered

    y_quantity = np.log1p(df_model["quantity"])
    model_q, X_test_q, preprocessor_q, features_q, num_filtered_q = train_and_save_model(y_quantity, "quantity")

    y_churn = df_model["churn"]
    model_c, X_test_c, preprocessor_c, features_c, num_filtered_c = train_and_save_model(y_churn, "churn")

    def plot_top_pdp(model, X_test, top_n, title, numeric_filtered):
        booster = model.named_steps["xgb"]
        importances = booster.feature_importances_
        feature_names_transformed = model.named_steps["preprocessor"].get_feature_names_out()

        raw_to_transformed = {
            raw: [f for f in feature_names_transformed if f.endswith(f"__{raw}")]
            for raw in numeric_filtered
        }

        top_indices = np.argsort(importances)[-top_n:][::-1]
        top_features_transformed = [feature_names_transformed[i] for i in top_indices if i < len(feature_names_transformed)]

        top_features_raw = []
        for raw, transformed_list in raw_to_transformed.items():
            if any(f in top_features_transformed for f in transformed_list):
                top_features_raw.append(raw)

        print("Top PDP raw features:", top_features_raw)

        if top_features_raw:
            PartialDependenceDisplay.from_estimator(model, X_test, features=top_features_raw, kind="average")
            plt.suptitle(title)
            plt.tight_layout()
            plt.show()
        else:
            print("No valid features for PDP.")

    print("\n--- PARTIAL DEPENDENCE PLOTS ---")
    plot_top_pdp(model_q, X_test_q, top_n=5, title="Partial Dependence Plots (Quantity)", numeric_filtered=num_filtered_q)
    plot_top_pdp(model_c, X_test_c, top_n=5, title="Partial Dependence Plots (Churn)", numeric_filtered=num_filtered_c)

    print("\n--- SHAP ANALYSIS (Quantity) ---")
    explainer_q = shap.Explainer(model_q.named_steps["xgb"])
    transformed_X_q = model_q.named_steps["preprocessor"].transform(X_test_q)
    feature_names_q = model_q.named_steps["preprocessor"].get_feature_names_out()
    shap_values_q = explainer_q(transformed_X_q)
    shap.summary_plot(shap_values_q, features=transformed_X_q, feature_names=feature_names_q)

    print("\n--- SHAP ANALYSIS (Churn) ---")
    explainer_c = shap.Explainer(model_c.named_steps["xgb"])
    transformed_X_c = model_c.named_steps["preprocessor"].transform(X_test_c)
    feature_names_c = model_c.named_steps["preprocessor"].get_feature_names_out()
    shap_values_c = explainer_c(transformed_X_c)
    shap.summary_plot(shap_values_c, features=transformed_X_c, feature_names=feature_names_c)

def predict_profit(df_new: pd.DataFrame, price: float, model_dir: str = "models") -> float:
    model_q = joblib.load(os.path.join(model_dir, "model_quantity_1.pkl"))
    model_c = joblib.load(os.path.join(model_dir, "model_churn_1.pkl"))

    df_new = df_new.copy()
    df_new["our_price"] = price
    X = df_new[model_q.named_steps["preprocessor"].feature_names_in_]

    demand = np.expm1(model_q.predict(X))
    churn = model_c.predict_proba(X)[:, 1]
    profit = (df_new["our_price"] - df_new["production_cost"]) * demand * (1 - churn)
    return profit.sum()

if __name__ == "__main__":
    df = pd.read_csv("data/data.csv")
    diagnose_price_quantity_relationship(df, num_regions=15, sample_size=2000)







