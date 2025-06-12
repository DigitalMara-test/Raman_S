# Agriculture Pricing Optimization (Bayesian Modeling with GPU)

This project focuses on building a **Bayesian hierarchical model** to estimate price elasticity of demand and optimize profit using **PyMC**, **JAX**, and **GPU acceleration (CUDA 12.2)** inside **WSL2**.

---
# 🌾 Crop Protection Analytics - Test Assignment Solution

This repository contains the full implementation of the solution for the **Advanced-Analytics Kynetec** test assignment focused on Price Analytics, Churn Prediction, and Strategic ROI Simulations in the crop protection domain.

---

## 📁 Repository Structure

```
Test_task_agriculture/
├── data/                       # Dataset folder (contains the CSV file)
├── models/                     # Trained XGBoost models for churn and quantity
├── task_description/           # (Optional) Task description files or notes
├── utils/                      # Helper utilities
├── Data_check.py               # Summary diagnostics and variable exploration
├── inspect_data.py             # Deep data quality profiling
├── Optimization.py             # Evolutionary optimization for profit maximization
├── preprocess_data.py          # Preprocessing logic for modeling
├── start_numpyro.py            # Hierarchical Bayesian model using NumPyro
├── Xgboost_train.py            # Training churn and quantity models with SHAP + PDP
├── README.md                   # Project overview (this file)
├── requirements.txt            # Python dependencies
└── research_data.ipynb         # Exploratory research notebook
```

---

## ✅ Implemented Tasks

### 1. 📉 Price Elasticity & Profit Maximization
- Developed both XGBoost and Bayesian models (`start_numpyro.py`) for demand modeling.
- Built an evolutionary optimizer (`Optimization.py`) using DEAP to maximize:
  ```
  Profit(p) = (p - c) * q(p) * (1 - churn(p))
  ```
- Constraints, frozen parameters, and input bounds are fully configurable.
- Partial Dependence Plots and SHAP for feature influence.

### 2. 🚪 Churn Prediction & Causal Drivers
- XGBoost-based classifier using extensive preprocessing (`Xgboost_train.py`).
- Feature selection by VIF, interpretation via SHAP.
- Estimated churn impact by state using expected gross margin loss.

### 3. 🌍 Geographic Price Optimization
- Regional analysis (`diagnose_price_quantity_relationship`):
  - Price variability
  - Correlation per region
  - State-based differentiation
- Future integration can include map visualizations.

### 4. 🔮 2025 ROI Simulation (Optional extension)
- Modular structure allows simulation of future year strategies using models:
  - General price increase in high `crop_yield` states
  - Reduction in `intensity` for regulatory-dense regions

---

## 🛠️ How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Train models
python Xgboost_train.py

# Step 3: Run optimization
python Optimization.py

# Optional: Run Bayesian model
python start_numpyro.py
```

---

## 🔍 Notes

- All scripts assume data is in `data/data.csv`.
- Trained models will be saved in the `models/` directory.
- Designed for extensibility: plug in constraints, custom objectives, or add regions easily.

---

## 🧠 Methodology Highlights

- DEAP evolutionary algorithm with frozen & bounded variable support.
- Multicollinearity check using VIF to reduce noise.
- SHAP values for interpreting model predictions.
- Cross-validation to ensure robustness.
- PDPs to visualize marginal effects.

---

## 🧾 Author Notes

This solution was implemented as part of the **Advanced-Analytics Kynetec** test assignment. It emphasizes clarity, 
explainability, and flexibility for real-world price and churn modeling in agriculture.---