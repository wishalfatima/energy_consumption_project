

# Household Energy Consumption Clustering and Forecasting

## Course Project

A data science project investigating whether grouping households by their electricity-consumption behavior can improve energy consumption forecasting.

**Author:** Wishal Fatima

---

## Project Overview

Energy consumption forecasting is important for energy management, demand planning, and efficient resource utilization.

This project analyzes daily electricity consumption data from approximately **17,547 households** in Austria. The main objective is to understand whether households with similar consumption behavior can be grouped into meaningful clusters and whether **cluster-specific forecasting models** can improve prediction accuracy compared with a single global forecasting model.

The project consists of two main components:

1. **Household clustering** based on consumption behavior
2. **Energy consumption forecasting** using global and cluster-specific machine learning models

The overall workflow is:

```text
2023 Electricity Consumption
            │
            ▼
   Data Exploration & Cleaning
            │
            ▼
     Feature Engineering
            │
            ▼
     Feature Normalization
            │
            ▼
        K-Means Clustering
            │
            ▼
     Household Cluster Profiles
            │
            ├───────────────┐
            ▼               ▼
     Global LightGBM   Cluster LightGBM
            │               │
            └───────┬───────┘
                    ▼
             2024 Forecasts
                    │
                    ▼
              MAE Evaluation
                    │
                    ▼
       Global vs Cluster Models
````

---

## Research Questions

The project investigates the following questions:

* Can households be grouped into meaningful consumption-based clusters?
* What characteristics distinguish the identified household groups?
* Does training separate forecasting models for different clusters improve forecasting accuracy?
* How does a cluster-based forecasting approach compare with a single global forecasting model?
* Can additional feature engineering and alternative forecasting strategies further improve performance?

---

## Dataset

The dataset contains daily electricity consumption for approximately **17,547 households**.

Two years of data are used:

| Dataset | Purpose                 | Time Period |
| ------- | ----------------------- | ----------: |
| 2023    | Training and clustering |    365 days |
| 2024    | Forecasting evaluation  |    366 days |

### Dataset Characteristics

| Property          |                         Value |
| ----------------- | ----------------------------: |
| Households        |                        17,547 |
| 2023 observations |        365 days per household |
| 2024 observations |        366 days per household |
| Data frequency    |                         Daily |
| Measurement       | Electricity consumption (kWh) |
| Training year     |                          2023 |
| Evaluation year   |                          2024 |

The 2024 data is reserved for evaluation and is not used to create the clustering features.

The raw dataset is not included in this repository because of its size.

---

## Data Exploration

Initial analysis was performed to understand the distribution and quality of the electricity consumption data.

The consumption distribution is strongly right-skewed. Most daily observations are relatively low, while a small number of observations have very high consumption values.

Some important statistics from the dataset include:

| Statistic                |        Value |
| ------------------------ | -----------: |
| Mean daily consumption   |    ~10.5 kWh |
| Median daily consumption |     ~7.4 kWh |
| 25th percentile          |     ~3.6 kWh |
| 75th percentile          |    ~13.4 kWh |
| 99th percentile          |    57.36 kWh |
| Maximum recorded value   | 1,051.74 kWh |
| Minimum                  |        0 kWh |

This analysis revealed substantial differences between households in terms of average consumption, variability, inactivity, and extreme consumption behavior.

---

# Feature Engineering

Household-level features were extracted from the 2023 consumption data.

The final feature set contains **13 features**.

### Original Features

| Feature             | Description                                        |
| ------------------- | -------------------------------------------------- |
| `mean_consumption`  | Average daily electricity consumption              |
| `std_consumption`   | Standard deviation of daily consumption            |
| `max_consumption`   | Maximum daily consumption                          |
| `min_consumption`   | Minimum daily consumption                          |
| `range_consumption` | Difference between maximum and minimum consumption |
| `cv`                | Coefficient of variation                           |
| `skewness`          | Asymmetry of the consumption distribution          |
| `zero_days`         | Number of days with zero consumption               |
| `zero_percentage`   | Percentage of zero-consumption days                |
| `p90`               | 90th percentile of consumption                     |
| `p10`               | 10th percentile of consumption                     |

Two additional log-transformed features were created to reduce the influence of highly skewed consumption values:

| Feature                | Description                 |
| ---------------------- | --------------------------- |
| `mean_consumption_log` | `log(1 + mean_consumption)` |
| `max_consumption_log`  | `log(1 + max_consumption)`  |

This resulted in a total of **13 clustering features**.

---

## Data Cleaning

Several data-quality issues were addressed before clustering.

### Constant households

A total of **154 households** had constant zero consumption throughout 2023.

These households produce undefined values for statistics such as coefficient of variation and skewness.

Missing skewness values were therefore handled through mean imputation after standardization.

### Skewness transformation

Consumption-related features are strongly right-skewed.

The transformations

```text
log(1 + mean_consumption)
log(1 + max_consumption)
```

were used to reduce the influence of extreme values while preserving the ordering of households.

### Feature normalization

The clustering features have different units and scales.

Therefore, `StandardScaler` was applied before clustering so that features with larger numerical ranges would not dominate the distance calculations.

---

# Household Clustering

## K-Means

K-Means clustering was used to group households according to their consumption characteristics.

The clustering was performed on the standardized household-level feature matrix.

The optimal number of clusters was investigated using clustering diagnostics such as the elbow analysis.

The final solution contained **four meaningful household groups**.

---

# Cluster Profiles

The resulting clusters represent substantially different consumption behaviors.

| Cluster | Profile               |  Share | Main Characteristics                          |
| ------: | --------------------- | -----: | --------------------------------------------- |
|       0 | Sparse / Low Activity |  ~2.5% | Low and irregular consumption, many zero days |
|       1 | Normal Consumers      | ~83.3% | Moderate, stable, and consistent consumption  |
|       2 | Inactive / Irregular  |  ~0.4% | Very low activity with many zero days         |
|       3 | High Consumers        | ~13.8% | High consumption and strong variability       |

### Cluster 0 — Sparse / Low Activity

This group contains households with relatively low and irregular consumption.

A large proportion of their days have zero consumption, suggesting intermittent activity.

---

### Cluster 1 — Normal Consumers

This is by far the largest group.

These households have moderate and relatively stable consumption patterns, making them the most representative group in the dataset.

---

### Cluster 2 — Inactive / Irregular

This is the smallest cluster.

These households have extremely low and irregular consumption with many zero-consumption days.

Their unusual behavior may represent inactive properties, irregularly used meters, or other special cases.

---

### Cluster 3 — High Consumers

This group contains households with significantly higher consumption and greater variability.

The higher variability makes these households more difficult to forecast accurately and motivates the use of dedicated cluster-specific forecasting models.

---

# Forecasting

The forecasting stage evaluates whether clustering households improves prediction accuracy.

Two main forecasting approaches were compared:

1. **Global forecasting**
2. **Cluster-based forecasting**

---

## Global Forecasting Model

A single **LightGBM** model was trained using data from all households.

The model learns general consumption patterns shared across the population.

The training data comes from 2023, while the final evaluation is performed on 2024.

---

## Cluster-Based Forecasting

Instead of training one model for every household, one LightGBM model was trained for each cluster.

Therefore:

```text
Cluster 0 → LightGBM Model 0
Cluster 1 → LightGBM Model 1
Cluster 2 → LightGBM Model 2
Cluster 3 → LightGBM Model 3
```

Each model was trained only on households belonging to its corresponding cluster.

This allows the forecasting model to specialize in the consumption behavior of a particular household group.

---

# Forecasting Features

The forecasting models use historical consumption together with calendar information.

The main feature set includes:

### Calendar Features

* `day_of_year`
* `day_of_week`
* `month`
* `weekend`

### Lag Features

* `lag_1`
* `lag_2`
* `lag_3`
* `lag_7`
* `lag_14`
* `lag_28`

These features capture short-term, weekly, and longer-term consumption patterns.

---

# Evaluation

Forecasting performance was evaluated using:

## Mean Absolute Error (MAE)

MAE is calculated as:

```text
MAE = mean(|actual - predicted|)
```

A lower MAE indicates better forecasting performance.

The evaluation compares predicted and actual electricity consumption over all **366 days of 2024**.

---

# Global vs Cluster-Based Forecasting

The initial forecasting experiment produced:

| Approach               |    MAE |
| ---------------------- | -----: |
| Global LightGBM        | 5.3218 |
| Cluster-based LightGBM | 5.2709 |

The cluster-based approach therefore achieved a lower overall MAE than the global model.

### Improvement

```text
Improvement = 5.3218 - 5.2709
            = 0.0509 MAE
```

This corresponds to an improvement of approximately:

```text
0.96%
```

The result suggests that grouping households according to consumption behavior can provide a small but measurable forecasting benefit.

---

# Per-Cluster Forecasting Performance

The cluster-specific models showed different forecasting difficulties.

| Cluster | Profile               | Validation MAE |
| ------: | --------------------- | -------------: |
|       0 | Sparse / Low Activity |         0.4225 |
|       1 | Normal Consumers      |         1.5772 |
|       2 | Inactive / Irregular  |         0.2778 |
|       3 | High Consumers        |         4.9902 |

The low MAE of Clusters 0 and 2 is partly explained by their near-zero consumption levels.

Cluster 3 is more difficult to forecast because high-consumption households also exhibit considerably greater variability.

---

# Forecasting Improvement Experiments

After the initial LightGBM results, two additional approaches were investigated.

## 1. XGBoost with Extended Features

XGBoost was tested using an expanded feature set including:

* rolling averages
* weekly differences
* cyclical seasonal features
* `day_of_year` sine/cosine encoding

The experiment used a sampled training dataset because of the large number of observations.

### Result

| Model                       |    MAE |
| --------------------------- | -----: |
| XGBoost + extended features | 6.1584 |

This was worse than the baseline LightGBM approach.

Therefore, the additional feature engineering and XGBoost model did not improve the final forecasting performance.

---

# 2. Recursive LightGBM Forecasting

A recursive forecasting strategy was also investigated.

Instead of predicting all future values independently, the model predicts one day at a time.

The predicted value is then added to the history and used to generate features for the next prediction.

The process is repeated for all 366 days of 2024.

### Recursive forecasting process

```text
Last 60 days of 2023
        │
        ▼
Generate features
        │
        ▼
Predict next day
        │
        ▼
Add prediction to history
        │
        ▼
Generate next features
        │
        ▼
Repeat for 366 days
```

The recursive model used 16 features including:

* calendar variables
* lag features
* rolling means
* trend features

This experiment was designed to investigate whether dynamically updating the history could improve long-horizon forecasting.

---

# Key Findings

### 1. Meaningful household groups can be identified

K-Means successfully separated households into groups with substantially different consumption characteristics.

The largest group represents normal consumers, while smaller groups capture sparse, inactive, and high-consumption households.

### 2. Cluster-specific forecasting can improve performance

The cluster-based LightGBM approach achieved a lower MAE than the single global model:

```text
Global LightGBM       → 5.3218 MAE
Cluster LightGBM     → 5.2709 MAE
```

This supports the hypothesis that households with different consumption behaviors can benefit from specialized forecasting models.

### 3. Different clusters have different forecasting difficulty

High-consumption households were considerably harder to predict than low-activity households.

This demonstrates that aggregate forecasting performance can hide substantial differences between household groups.

### 4. XGBoost did not improve the baseline

The XGBoost experiment achieved an MAE of 6.1584, which was worse than the original LightGBM approaches.

This demonstrates that adding more features or using a different gradient-boosting algorithm does not automatically improve forecasting performance.

### 5. Cluster-based modeling is promising

Although the improvement in overall MAE is relatively small, the results demonstrate the potential value of segmentation before forecasting.

Rather than assuming every household follows the same consumption process, clustering allows models to specialize in groups with similar behavior.

---

# Repository Structure

```text
energy_consumption_project/
│
├── data/
│   └── Raw datasets
│
├── notebooks/
│   ├── Data exploration
│   ├── Feature engineering
│   ├── Clustering
│   ├── Forecasting
│   └── Forecasting improvement experiments
│
├── outputs/
│   ├── Cluster assignments
│   ├── Clustering results
│   ├── Forecasting results
│   ├── Model summaries
│   ├── Evaluation files
│   └── Visualizations
│
└── README.md
```

---

# Main Outputs

The `outputs/` directory contains the main results generated during the project.

Examples include:

```text
cluster_assignments.csv
cluster_results.csv
final_results.csv
optimized_results.csv

cluster_patterns.png
cluster_pca_final.png
final_comparison.png
optimal_k_analysis.png

clustering_features_bar_charts.png
clustering_features_grouped_bars.png
clustering_features_horizontal_bars.png

global_recursive_results.png
per_cluster_mae_recursive.png
```

The output files provide both numerical results and visualizations of the clustering and forecasting experiments.

---

# Reproducibility

## Requirements

The project uses Python and common data science and machine learning libraries.

Main dependencies include:

```text
Python 3.x

pandas
numpy
scikit-learn
lightgbm
xgboost
matplotlib
seaborn
scipy
jupyter
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not yet present, it should be added to the repository.

---

## Running the Project

1. Clone the repository:

```bash
git clone https://github.com/wishalfatima/energy_consumption_project.git
```

2. Enter the project directory:

```bash
cd energy_consumption_project
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Place the required datasets in the `data/` directory.

5. Run the notebooks in the following general order:

```text
1. Data Exploration
        ↓
2. Feature Engineering
        ↓
3. Clustering
        ↓
4. Cluster Analysis
        ↓
5. Global Forecasting
        ↓
6. Cluster-Based Forecasting
        ↓
7. Forecasting Improvements
```

---

# Data Availability

The raw electricity consumption dataset is not included in this repository because of its size and data-access restrictions.

The project was developed using the electricity consumption data provided for the course project.

If you have access to the original dataset, place the required files inside:

```text
data/
```

before running the notebooks.

---

# Limitations

Several limitations should be considered when interpreting the results.

### 1. Household clustering is based on 2023 behavior

The clusters are created using historical consumption characteristics from 2023.

Changes in household behavior during 2024 are therefore not incorporated into the clustering stage.

### 2. Cluster-based improvement is relatively small

The overall improvement from the initial global model to the cluster-based model is modest.

Therefore, clustering should be viewed as a potentially useful segmentation strategy rather than a guaranteed large improvement.

### 3. Small clusters

Some clusters contain very few households.

Their low MAE values should therefore be interpreted carefully because their behavior may not be representative of the overall population.

### 4. Extreme consumption values

The dataset contains a small number of extremely high consumption observations.

Although feature transformations and standardization were used during clustering, these extreme values can still influence the forecasting models.

---

# Conclusion

This project demonstrates a complete pipeline for household electricity consumption analysis, clustering, and forecasting.

The main finding is that **cluster-based forecasting slightly outperformed a single global LightGBM model**, with MAE improving from:

```text
5.3218 → 5.2709
```

The clustering analysis also revealed four distinct household behavior profiles:

* Sparse / Low Activity
* Normal Consumers
* Inactive / Irregular
* High Consumers

The results suggest that electricity consumers are heterogeneous and that recognizing these behavioral differences can be useful for forecasting.

At the same time, the experiments show that more complex models do not necessarily produce better results. The XGBoost experiment with additional features performed worse than the baseline LightGBM approach.

Overall, the project provides an end-to-end example of how **data exploration, behavioral feature engineering, clustering, and machine learning forecasting** can be combined to analyze household energy consumption.

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* XGBoost
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Author

**Wishal Fatima**

University of Vienna

Course Project — Household Energy Consumption Clustering and Forecasting

```

