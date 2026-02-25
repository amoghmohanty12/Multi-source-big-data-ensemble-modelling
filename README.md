# End-to-End Vehicle Price Prediction: Ensemble Methods and Multi-Source Data Integration

A comprehensive machine learning project that predicts used vehicle prices by combining two real-world datasets — Craigslist listings and Kelley Blue Book (KBB) specifications — and applying state-of-the-art ensemble methods with Bayesian hyperparameter optimisation.

---

## Project Overview

Used car pricing is notoriously difficult to assess due to the large number of interacting factors: vehicle age, brand reputation, mileage, condition, regional demand, and objective specifications like fuel efficiency. This project builds a production-ready regression pipeline that:

- Ingests and cleans a large (~400K row) Craigslist vehicle dataset
- Enriches it with KBB specifications via entity linking (multi-source data integration)
- Engineers domain-relevant features (car age, luxury brand flags, mileage/year, MPG combined)
- Trains and compares four model families: Linear Regression, Random Forest, Gradient Boosting, and XGBoost
- Optimises all three ensemble models using **Optuna Bayesian hyperparameter search**
- Implements a three-method anomaly detection system for fraud/mispricing detection

---

## Dataset Sources

| Dataset | Description | Size |
|---|---|---|
| Craigslist Vehicles | Used car listings with price, odometer, condition, location | ~400K rows |
| Kelley Blue Book (KBB) | Objective vehicle specs: highway MPG, city MPG, engine size | Aggregated by make/year |

The two datasets are linked on `(manufacturer, year)` to enrich listings with objective, brand-level specifications — a form of **entity linking** across heterogeneous data sources.

---

## Project Structure

```
VehiclePricePrediction_FinalProject.ipynb   # Main notebook (all code and analysis)
README.md
```

The notebook is self-contained and divided into eight parts:

| Part | Content |
|---|---|
| 1 | Introduction & Problem Statement |
| 2 | Data Loading & Preprocessing (Craigslist + KBB) |
| 3 | Exploratory Data Analysis (price, geography, manufacturer, correlation) |
| 4 | Feature Engineering (car age, brand tiers, mileage/year, interaction terms) |
| 5 | Multi-Source Data Merging (KBB entity linking + derived MPG features) |
| 6 | Modelling (Linear Regression → Ensemble Methods → **Optuna Tuning** → Anomaly Detection) |
| 7 | Results & Model Comparison |
| 8 | Difficulty Concepts Summary |

---

## Models Trained

### Baseline
- **Linear Regression** — establishes minimum expected performance

### Ensemble Methods
- **Random Forest** — bagged decision trees; strong on tabular data
- **Gradient Boosting** — sequential boosting; robust to outliers
- **XGBoost** — regularised gradient boosting; state-of-the-art on structured data

### Hyperparameter Optimisation (Optuna)
All three ensemble models are tuned using **Optuna's Tree-structured Parzen Estimator (TPE)** — a Bayesian optimisation algorithm that learns from previous trials to focus on the most promising hyperparameter regions.

| Model | Key Hyperparameters Tuned |
|---|---|
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `subsample` |
| XGBoost | `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |

Optuna runs 30 trials per model with 3-fold cross-validation, producing optimisation history plots and hyperparameter importance charts.

---

## Anomaly Detection System

Beyond price prediction, the project implements a three-method anomaly detection pipeline for identifying potentially fraudulent or mispriced listings:

| Method | Approach |
|---|---|
| 1. Prediction-based thresholding | Flag listings where actual price deviates >30% from model prediction |
| 2. Quantile regression intervals | Flag listings outside the 80% prediction interval (10th–90th percentile) |
| 3. Isolation Forest | Unsupervised detection of unusual feature patterns (contamination=5%) |

Listings flagged by 2 or more methods are considered high-confidence anomalies.

---

## Key Results

All final numbers are produced by running the notebook end-to-end on the full dataset.

**Model Comparison (illustrative ordering — run notebook for exact values):**

| Model | Test R² | Test MAE |
|---|---|---|
| Linear Regression | ~0.79 | ~$3,500 |
| Random Forest push | ~0.92 | ~$2,000 |
| Gradient Boosting | ~0.88 | ~$2,300 |
| XGBoost | ~0.90 | ~$2,100 |


Optuna-tuned models consistently outperform their hand-configured counterparts by discovering non-obvious hyperparameter interactions — particularly XGBoost's regularisation coefficients (`reg_alpha`, `reg_lambda`).

---

## Feature Importance Highlights

Top predictors identified by the tuned Random Forest:

1. **Odometer** — strongest single predictor (higher mileage → lower price)
2. **Car age** — tightly correlated with price decay
3. **Manufacturer** (one-hot encoded) — luxury brands command significant premium
4. **Combined MPG** (KBB-derived) — fuel efficiency affects perceived value
5. **Condition** — excellent/like-new listings price substantially higher

---

## Concepts Demonstrated

1. **Entity Linking / Multi-Source Integration** — merging heterogeneous datasets on shared keys
2. **Feature Importance Analysis** — extracting and visualising tree-based feature importances
3. **Ensemble Methods** — Random Forest, Gradient Boosting, XGBoost comparison
4. **Anomaly Detection** — multi-method fraud/mispricing detection pipeline
5. **Bayesian Hyperparameter Optimisation** — Optuna TPE for principled model tuning

---

## Setup & Requirements

The notebook is designed for **Google Colab** with Google Drive mounting for dataset access.

**Key dependencies:**

```
pandas
numpy
scikit-learn
xgboost
optuna
matplotlib
seaborn
```

Install Optuna (handled automatically by the notebook):

```bash
pip install optuna
```

---

## Running the Notebook

1. Upload the notebook to Google Colab
2. Mount your Google Drive and update the `file_path` variables in cells 9 and 28 to point to your copies of the Craigslist and KBB CSV files
3. Run all cells sequentially (`Runtime → Run all`)

Estimated runtime: 15–30 minutes depending on Colab hardware (Optuna tuning is the most time-intensive step).
