 Energy Consumption Forecasting: Time Series Clustering & Prediction

Project Overview

This project analyzes and forecasts household energy consumption using **time series clustering** and **machine learning forecasting models**. The dataset contains daily energy consumption for **17,547 households** over two years (2023 for training, 2024 for testing).

Key Findings:
- Clustering households into 3 groups revealed distinct consumption patterns
- XGBoost achieved the best forecasting performance (MAE: 4.45)
-Cluster-based models showed NO improvement over global model
- A single global forecasting model is sufficient for this dataset

---

Project Tasks

| Task | Description | Status |
|------|-------------|--------|
| **Task 1** | Clustering households based on consumption patterns 
| **Task 2** | Forecasting (Global vs Cluster-based models) 
| **Task 3** | Evaluation using MAE on 2024 data 

---

 Dataset

| Dataset | Shape | Description |
|---------|-------|-------------|
| `sample_23.csv` | 17,547 × 365 | Training data (2023) |
| `sample_24.csv` | 17,547 × 366 | Testing data (2024 - leap year) |

Key Statistics:
- Normal consumption: 7-15 units/day
- Extreme outliers found:** 123 households (0.7%) with values up to 1051 units
- Outliers handled: Capped at 99.9th percentile (52.72)

---

Methodology

1. Data Preprocessing
- Handled extreme outliers (capping at 99.9th percentile)
- Extracted 25 features per household for clustering
- No data leakage: Used ONLY 2023 data for training

 2. Clustering (k=3)
- Algorithm: K-means
- Features used: 11 features (mean, std, trend, weekend patterns, etc.)
- Optimal k: 3 clusters based on silhouette score

3. Forecasting
4. Optimization Improvement

energy_consumption_project/
│
├── notebooks/
│ ├── 01_data_exploration.ipynb # Initial data analysis
│ ├── 02_feature_engineering.ipynb # Feature extraction
│ ├── 03_clustering_new.ipynb # K-means clustering (k=3)
│ ├── 04_forecasting_xgboost.ipynb # XGBoost forecasting
│ └── 05_forecasting_lightgbm.ipynb # LightGBM forecasting
│
├── data/
│ ├── raw/  Original CSV files
│ └── processed/  Cleaned data & results
│
├── research_diary/ Daily documentation
│
└── README.md 


How to Run

 1. Clone the Repository
```bash
git clone https://github.com/wishalfatima/energy_consumption_project.git
cd energy_consumption_project
2. Install Dependencies
bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
3. Run Jupyter Notebooks
bash
jupyter notebook
Run notebooks in order: 01_ → 02_ → 03_ → 04_ → 05_ → 


Key Insights
Clustering doesn't improve forecasting for this dataset

Simple global model performs as well as complex cluster-specific models

XGBoost outperformed LightGBM slightly (but both similar)

Only 0.7% of households were extreme outliers (123 out of 17,547)

Technologies Used
Tool	Purpose
Python	Main programming language
Pandas/NumPy	Data manipulation
Scikit-learn	Clustering, scaling
XGBoost	Forecasting model
LightGBM	Forecasting model
Matplotlib/Seaborn	Visualizations
Jupyter	Interactive notebooks










