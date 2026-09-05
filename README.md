# HFT Price Prediction — Regression

Machine Learning regression project for predicting the **future trade price after 5 seconds** using High-Frequency Trading (HFT) data.

## 🎯 Objective

Convert the original **classification problem into a regression problem** to predict the actual future trade price after 5 seconds.

## 🔄 Changes Made

* Classification → Regression
* Target → `future_trade_price_5s`
* Models → Random Forest Regressor + HistGradientBoosting Regressor
* Feature Selection → Mutual Information + SelectKBest
* Evaluation → MAE, RMSE, R²
* Ensemble → Average predictions from both models
* Data Split → 80% Training / 20% Testing (chronological)

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

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the modelling pipeline:

```bash
python modelling_pipeline.py
```

## 📁 Main Files

```text
HFT-price-prediction-ml/
│
├── data.csv
├── modelling_pipeline.py
├── modelling_pipeline_backup.py
├── requirements.txt
├── metrics.json
├── feature_selector.joblib
├── hist_gradient_boosting_regressor.joblib
├── ASSIGNMENT_MODIFICATION.txt
└── README.md
```

## 📌 Project

**HFT Price Prediction — Modified Regression Version**
