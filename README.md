# Household Energy Consumption Clustering & Forecasting

A machine learning project for analyzing household electricity consumption patterns, clustering households based on their consumption behavior, and evaluating whether cluster-specific forecasting models improve energy consumption predictions.

---

## Project Overview

Energy consumption forecasting is important for energy management, planning, and understanding consumer behavior.

This project investigates two connected questions:

1. **Can households be grouped into meaningful consumption-behavior clusters?**
2. **Can cluster-specific forecasting models improve prediction accuracy compared with a single global forecasting model?**

The project uses daily electricity consumption data from approximately **17,547 households** in Austria.

The workflow consists of:

```text
2023 Electricity Consumption
          │
          ▼
   Data Exploration
          │
          ▼
  Feature Engineering
          │
          ▼
   Feature Selection
          │
          ▼
      K-Means
     Clustering
          │
          ▼
 Cluster Interpretation
          │
          ▼
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
Global Forecasting   Cluster Forecasting
 │                       │
 └───────────┬───────────┘
             ▼
       2024 Evaluation
             │
             ▼
        MAE Comparison




Dataset

The dataset contains daily electricity consumption for 17,547 households.

Property	Value
Households	17,547
Training period	2023
Evaluation period	2024
2023 observations	365 days
2024 observations	366 days
Measurement	Daily electricity consumption
Unit	kWh

The 2023 data is used for feature engineering, clustering, and model training, while 2024 is reserved for forecasting evaluation.

The 2024 dataset is therefore not used to determine the household clusters.

1. Exploratory Data Analysis

The first stage investigates the distribution and characteristics of household electricity consumption.

The consumption data is strongly right-skewed.

Typical daily consumption is concentrated in the lower range, while a small number of households show extremely high values.

Key statistics
Statistic	Value
Mean daily consumption	~10.5 kWh
Median	~7.4 kWh
25th percentile	~3.6 kWh
75th percentile	~13.4 kWh
99th percentile	57.36 kWh
Maximum	1,051.74 kWh
Minimum	0 kWh

The extreme maximum highlights the presence of highly unusual or high-consumption households.

2. Feature Engineering

Instead of clustering directly on 365 daily consumption values, statistical features were extracted from each household's 2023 consumption profile.

A total of 13 features were created.

Original Features
Feature	Description
mean_consumption	Average daily consumption
std_consumption	Day-to-day variability
max_consumption	Highest daily consumption
min_consumption	Lowest daily consumption
range_consumption	Difference between maximum and minimum
cv	Coefficient of variation
skewness	Distribution asymmetry
zero_days	Number of zero-consumption days
zero_percentage	Percentage of zero-consumption days
p90	90th percentile
p10	10th percentile

Two additional log-transformed features were created:

Feature	Purpose
mean_consumption_log	Reduces right-skew in mean consumption
max_consumption_log	Reduces right-skew in maximum consumption

This resulted in 13 features per household.

3. Data Cleaning

Several data-quality issues were investigated before clustering.

Constant-zero households

A total of 154 households had zero consumption throughout the entire year.

These households resulted in undefined statistical measures such as skewness and coefficient of variation.

Missing feature values were handled through mean imputation after standardization.

After preprocessing:

Check	Before	After
NaN in skewness	154	0
NaN in other features	0	0
Total NaN	154	0
Infinite values	0	0

The final feature matrix contained:

17,547 households × 6 selected clustering features
4. Feature Selection

Using every available feature can introduce redundancy and correlated information.

Therefore, six features were selected for clustering:

mean_consumption_log
std_consumption
cv
zero_percentage
skewness
range_consumption

The selected features capture:

consumption level
variability
relative volatility
inactivity
distribution shape
overall consumption range

All clustering features were standardized using StandardScaler.

5. Household Clustering
K-Means

K-Means was selected as the main clustering algorithm because it is computationally efficient and suitable for clustering a large number of households.

The number of clusters was evaluated from:

k = 2 ... 14

Three metrics were considered:

Inertia / Elbow Method
Silhouette Score
Davies-Bouldin Index
Optimal number of clusters

The analysis selected:

k = 4

The choice was supported by the combination of the three clustering metrics.

k	Inertia	Silhouette	Davies-Bouldin
2	80,348	0.4650	0.9958
3	57,851	0.5414	0.8541
4	51,437	0.5456	0.7551
5	47,288	0.2295	1.1660
6	35,283	0.2957	0.9702
7	38,416	0.2376	1.0548
8	36,591	0.2222	1.1205
9	26,621	0.2184	1.0383
10	27,131	0.2316	1.1347
11	22,967	0.2295	1.0255
12	22,287	0.2205	1.0188
13	20,235	0.2542	0.9984
14	18,736	0.2156	1.0194

The strongest combination of separation and compactness occurred at k = 4.

6. Final Cluster Profiles

The final K-Means model produced four household groups.

Cluster	Households	Share	Interpretation
0	444	2.5%	Sparse / Low Activity
1	14,620	83.3%	Normal Consumers
2	66	0.4%	Ghost / Inactive Meters
3	2,417	13.8%	High Consumers
Cluster 0 — Sparse / Low Activity

Small group of households with low and irregular consumption and many zero-consumption days.

These households may represent intermittently occupied properties or low-activity meters.

Cluster 1 — Normal Consumers

The dominant cluster, containing approximately 83% of households.

These households have moderate and relatively stable consumption and represent the typical consumption pattern in the dataset.

Cluster 2 — Ghost / Inactive Meters

A very small group characterized by extremely low consumption and a high proportion of zero days.

Cluster 3 — High Consumers

Households with significantly higher consumption and substantially greater variability.

This group benefits from dedicated forecasting models because its behavior differs considerably from the majority of households.

7. Forecasting

After clustering, the project evaluates whether separate forecasting models for each cluster can improve prediction accuracy.

Two forecasting strategies were investigated:

Global forecasting
Cluster-based forecasting
Feature Engineering for Forecasting

The consumption data was converted from wide format into a time-series format.

Calendar features:

day_of_year
day_of_week
month
weekend

Lag features:

lag_1
lag_2
lag_3
lag_7
lag_14
lag_28

These features capture short-term, weekly, and longer-term consumption patterns.

8. Global Forecasting Model

A single LightGBM model was trained using all households.

The model learns general electricity consumption patterns across the complete population.

Feature importance showed that:

day_of_year was highly influential
lag_1 was an important short-term predictor
lag_7 and lag_14 captured weekly patterns
weekend contributed comparatively little because weekly behavior was already represented by day_of_week
9. Cluster-Based Forecasting

Instead of using one model for every household, a separate LightGBM model was trained for each cluster.

Cluster 0 → LightGBM Model 0
Cluster 1 → LightGBM Model 1
Cluster 2 → LightGBM Model 2
Cluster 3 → LightGBM Model 3

All models use the same forecasting methodology so that the comparison reflects the effect of clustering rather than differences in model architecture.

Cluster models
Cluster	Households	Model
0	444	Dedicated LightGBM
1	14,620	Dedicated LightGBM
2	66	Dedicated LightGBM
3	2,417	Dedicated LightGBM

No cluster required fallback to the global model.

10. Forecasting Evaluation

The models were evaluated on the complete 2024 dataset.

The evaluation metric is:

Mean Absolute Error (MAE)
MAE=
n
1
	​

i=1
∑
n
	​

∣y
i
	​

−
y
^
	​

i
	​

∣

Lower MAE indicates better forecasting accuracy.

11. Baseline Forecasting Results

The initial comparison produced:

Model	MAE
Global LightGBM	5.3218
Cluster-based LightGBM	5.2709

The cluster-based approach therefore achieved a small improvement over the global model.

This suggests that separating households according to consumption behavior can help the forecasting model learn more homogeneous patterns.

12. Forecasting Improvement Experiments

Two additional approaches were investigated.

Experiment 1 — XGBoost

An XGBoost model was tested with additional features including:

rolling mean
weekly difference
cyclical day-of-year encoding

The experiment used a sampled training set because of the large dataset size.

Result
Model	MAE
XGBoost + Extended Features	6.1584

This was worse than the LightGBM baseline, so the approach was not selected as the final model.

13. Recursive LightGBM Forecasting

A recursive forecasting strategy was then investigated.

Instead of predicting all future values independently, predictions are generated sequentially:

2023 history
     ↓
Predict Day 1
     ↓
Add prediction to history
     ↓
Predict Day 2
     ↓
Add prediction to history
     ↓
...
     ↓
Predict Day 366

A rolling history of the last 60 days was used.

The recursive model used:

Calendar features
day_of_year
day_of_week
month
weekend
Lag features
1
2
3
7
14
28
30 days
Rolling features
7-day mean
14-day mean
30-day mean
Trend features
diff_7
trend_7

Predictions were clipped to non-negative values.

14. Main Findings
Finding 1 — Household behavior can be meaningfully clustered

K-Means identified four distinct consumption groups ranging from inactive households to high-consuming households.

Finding 2 — Cluster-specific forecasting can improve accuracy

The cluster-based LightGBM model achieved:

5.2709 MAE

compared with:

5.3218 MAE

for the global model.

Finding 3 — Household clusters have very different forecasting difficulty

The normal-consumer cluster is relatively stable, while high-consumption households exhibit substantially greater variability.

Finding 4 — XGBoost did not improve the baseline

The tested XGBoost configuration achieved an MAE of 6.1584, which was worse than the LightGBM baseline.

Finding 5 — Recursive forecasting is promising

Recursive forecasting allows predictions to be incorporated into subsequent predictions and provides a more dynamic representation of future consumption.

15. Project Structure
energy_consumption_project/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_forecasting.ipynb
│   └── 05_forecasting_improved.ipynb
│
├── outputs/
│   ├── clustering/
│   ├── forecasting/
│   └── figures/
│
├── README.md
└── requirements.txt
16. Technologies
Python
Pandas
NumPy
Scikit-learn
LightGBM
XGBoost
Matplotlib
Seaborn
Jupyter Notebook
17. Reproducibility
1. Clone the repository
git clone https://github.com/wishalfatima/energy_consumption_project.git
cd energy_consumption_project
2. Install dependencies
pip install -r requirements.txt
3. Add the dataset

The raw electricity consumption data is not included in this repository because of its size and data-sharing restrictions.

Place the 2023 and 2024 datasets inside:

data/
4. Run the notebooks

Run the notebooks in the following order:

01_data_analysis.ipynb
        ↓
02_feature_engineering.ipynb
        ↓
03_clustering.ipynb
        ↓
04_forecasting.ipynb
        ↓
05_forecasting_improved.ipynb
18. Outputs

The project produces:

consumption distribution visualizations
feature statistics
feature correlation analysis
optimal-k analysis
PCA cluster visualization
cluster profiles
cluster assignments
feature importance plots
global forecasting results
cluster forecasting results
recursive forecasting results
MAE comparisons
19. Key Result

The central finding of the project is:

Grouping households according to their consumption behavior and training dedicated forecasting models can provide better forecasting accuracy than using a single global model.

The initial cluster-based LightGBM model improved MAE from:

Global:   5.3218
Cluster:  5.2709

showing that localized forecasting can capture differences between household consumption patterns.

Author

Wishal Fatima

University of Vienna

Computer Science

Project Context

This project was developed as part of a university course project using real-world household electricity consumption data.


---

