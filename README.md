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
