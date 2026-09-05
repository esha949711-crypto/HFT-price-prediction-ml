# HFT Price Prediction — Regression

Machine Learning project for predicting the **future trade price after 5 seconds** using High-Frequency Trading (HFT) data.

## 🎯 Objective

Convert the original **classification problem into a regression problem** to predict an actual future trade price.

## 🔄 Changes Made

* Classification → Regression
* Target → `future_trade_price_5s`
* Models → Random Forest Regressor + HistGradientBoosting Regressor
* Feature Selection → Mutual Information + SelectKBest
* Evaluation → MAE, RMSE, R²
* Ensemble → Average predictions from both models
* Split → 80% Training / 20% Testing (chronological)

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

## 📊 Results

| Metric |     Model |  Baseline |
| ------ | --------: | --------: |
| MAE    | 43.137278 | 39.949631 |
| RMSE   | 48.903659 | 49.952767 |
| R²     | -2.807378 | -2.972486 |

**Selected Features:** 25
**Training Rows:** 87,193
**Testing Rows:** 21,799

## 🚀 How to Run

```bash
pip install -r requirements.txt
python modelling_pipeline.py
```

## 📁 Main Files

```text
data.csv
modelling_pipeline.py
README.md
requirements.txt
random_forest_regressor.joblib
hist_gradient_boosting_regressor.joblib
feature_selector.joblib
features.txt
metrics.json
```

## 👩‍💻 Project
**HFT Price Prediction — Modified Regression Version**
