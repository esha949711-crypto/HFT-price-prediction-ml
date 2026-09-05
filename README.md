# 📈 HFT Price Prediction — Regression

> **Machine Learning regression project for predicting future trade prices after 5 seconds using High-Frequency Trading (HFT) data.**

---

## 🎯 Project Task

The task is to modify an existing **HFT price movement classification project** into a **regression-based prediction system**.

Instead of predicting whether the price will move **Up or Down**, the modified project predicts the **actual future trade price after 5 seconds**.

### Original Task

**Classification:** Predict short-term price movement/class.

### Modified Task

**Regression:** Predict the numerical value of:

```text
future_trade_price_5s
```

---

## 🎯 Objective

Build a Machine Learning regression pipeline that uses HFT market data to predict the **future trade price 5 seconds ahead**.

The project focuses on:

* Preparing HFT data for regression
* Selecting the most useful features
* Training regression models
* Combining model predictions
* Evaluating prediction performance

---

## 🔄 Modifications Made

| Original               | Modified                  |
| ---------------------- | ------------------------- |
| Classification         | Regression                |
| Price movement/class   | Future numerical price    |
| Classification target  | `future_trade_price_5s`   |
| Classification models  | Regression models         |
| Classification metrics | MAE, RMSE, R²             |
| Random split           | Chronological 80/20 split |

### Additional Improvements

* **Feature Selection:** Mutual Information + SelectKBest
* **Models:** Random Forest Regressor + HistGradientBoosting Regressor
* **Ensemble:** Average predictions from both models
* **Time-Aware Split:** 80% training / 20% testing

---

## 🔄 Machine Learning Workflow

```text
HFT Market Data
       ↓
Data Preparation
       ↓
Feature Selection
       ↓
Chronological Train/Test Split
       ↓
Regression Models
       ↓
Ensemble Predictions
       ↓
5-Second Future Price
       ↓
MAE / RMSE / R² Evaluation
```

---

## 🤖 Machine Learning Models

### 🌲 Random Forest Regressor

Uses multiple decision trees and combines their predictions to estimate the future trade price.

### ⚡ HistGradientBoosting Regressor

A gradient boosting regression model that builds trees sequentially to improve prediction performance.

### 🤝 Ensemble

Predictions from the regression models are averaged to produce the final prediction.

---

## 🔍 Feature Selection

The project uses:

**Mutual Information + SelectKBest**

to identify the most informative features for predicting the target.

### Selected Features

**25 features**

---

## 📊 Dataset Split

Because HFT data is time-dependent, the dataset is split **chronologically** rather than randomly.

```text
Older Data                         Newer Data
     │                                  │
     ├──────── 80% Training ────────────┤── 20% Testing ──┤
```

### Dataset Information

* **Training Rows:** 87,193
* **Testing Rows:** 21,799
* **Selected Features:** 25

---

## 📈 Results

| Metric   |     Model |  Baseline |
| -------- | --------: | --------: |
| **MAE**  | 43.137278 | 39.949631 |
| **RMSE** | 48.903659 | 49.952767 |
| **R²**   | -2.807378 | -2.972486 |

### Evaluation Metrics

* **MAE:** Measures the average absolute difference between actual and predicted prices.
* **RMSE:** Measures prediction error while giving greater weight to larger errors.
* **R²:** Measures how well the model explains the variation in the target.

> **Note:** The current R² score is negative, indicating that the model's predictive performance is weak relative to the baseline. The results are reported transparently as part of the project evaluation.

---

## 🛠️ Technologies

* 🐍 Python
* 🐼 Pandas
* 🔢 NumPy
* 🤖 Scikit-learn
* 💾 Joblib

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/esha949711-crypto/HFT-price-prediction-ml.git
```

### 2. Open the Project Directory

```bash
cd HFT-price-prediction-ml
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Model Pipeline

```bash
python modelling_pipeline.py
```

---

## 📁 Project Structure

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

---

## 📌 Key Project Details

| Component              | Details                              |
| ---------------------- | ------------------------------------ |
| **Problem Type**       | Regression                           |
| **Domain**             | High-Frequency Trading               |
| **Prediction Horizon** | 5 seconds                            |
| **Target**             | `future_trade_price_5s`              |
| **Feature Selection**  | Mutual Information + SelectKBest     |
| **Models**             | Random Forest + HistGradientBoosting |
| **Ensemble**           | Average Predictions                  |
| **Train/Test Split**   | 80% / 20% Chronological              |
| **Evaluation**         | MAE, RMSE, R²                        |
| **Selected Features**  | 25                                   |

---

## 📚 Learning Outcome

This project provided practical experience with a complete Machine Learning workflow, including:

* Data preparation
* Feature selection
* Regression modelling
* Ensemble prediction
* Time-based train/test splitting
* Model evaluation
* Saving trained ML artifacts
* Documenting a Machine Learning project on GitHub

---

## 👩‍💻 Project

**HFT Price Prediction — Modified Regression Version**

Built as a Machine Learning project to demonstrate the conversion of an HFT classification task into a regression-based future price prediction task.
