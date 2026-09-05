import json
import os
from bisect import bisect_left

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


NUMERIC_COLUMNS = [
    "bid_price", "bid_qty", "ask_price", "ask_qty", "trade_price",
    "sum_trade_1s", "bid_advance_time", "ask_advance_time", "last_trade_time"
]


def preprocessing(data):
    """Convert data types and sort the HFT observations chronologically."""
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.sort_values("timestamp").reset_index(drop=True)


def fill_null(data):
    """Fill missing trading activity values using the original project logic."""
    data = data.copy()
    data["sum_trade_1s"] = data["sum_trade_1s"].fillna(0)

    last_trade = []
    previous_time = None
    previous_last_trade = np.nan

    for timestamp, value in zip(data["timestamp"], data["last_trade_time"]):
        if pd.isna(value):
            if previous_time is not None:
                interval = (timestamp - previous_time).total_seconds()
                if interval <= 1:
                    value = previous_last_trade + interval
                else:
                    value = np.nan
        last_trade.append(value)
        previous_time = timestamp
        previous_last_trade = value

    data["last_trade_time"] = last_trade
    return data


def create_regression_target(data, horizon_seconds=5):
    """
    Create a numerical target: the first trade price observed at least
    `horizon_seconds` after the current timestamp.

    This changes the original classification task into a price-regression task.
    """
    data = data.copy()
    timestamps = data["timestamp"].astype("int64").to_numpy()
    prices = data["trade_price"].to_numpy()

    future_ns = timestamps + int(horizon_seconds * 1e9)
    future_index = np.searchsorted(timestamps, future_ns, side="left")

    target = np.full(len(data), np.nan)
    valid = future_index < len(data)
    target[valid] = prices[future_index[valid]]

    data["future_trade_price_5s"] = target
    return data


def engineer_features(data):
    """Build market microstructure, lag and rolling features."""
    data = data.copy()

    data["spread"] = data["ask_price"] - data["bid_price"]
    data["mid_price"] = (data["ask_price"] + data["bid_price"]) / 2
    data["price_to_mid"] = data["trade_price"] - data["mid_price"]
    data["bid_ask_qty_total"] = data["bid_qty"] + data["ask_qty"]
    data["bid_ask_qty_diff"] = data["ask_qty"] - data["bid_qty"]
    data["order_imbalance"] = (
        (data["bid_qty"] - data["ask_qty"]) /
        (data["bid_qty"] + data["ask_qty"] + 1e-9)
    )

    data["trade_price_compare"] = 0
    data.loc[data["trade_price"] <= data["bid_price"], "trade_price_compare"] = -1
    data.loc[data["trade_price"] >= data["ask_price"], "trade_price_compare"] = 1

    base_cols = [
        "bid_price", "bid_qty", "ask_price", "ask_qty", "trade_price",
        "sum_trade_1s", "bid_advance_time", "ask_advance_time",
        "last_trade_time", "spread", "mid_price", "price_to_mid",
        "bid_ask_qty_total", "bid_ask_qty_diff", "order_imbalance",
        "trade_price_compare"
    ]

    for col in base_cols:
        data[f"{col}_diff"] = data[col].diff()
        data[f"{col}_lag_1"] = data[col].shift(1)
        data[f"{col}_lag_2"] = data[col].shift(2)
        data[f"{col}_lag_5"] = data[col].shift(5)

    rolling_cols = [
        "trade_price", "spread", "mid_price", "bid_qty", "ask_qty",
        "sum_trade_1s", "order_imbalance"
    ]
    for col in rolling_cols:
        data[f"{col}_rolling_mean_5"] = data[col].rolling(5).mean()
        data[f"{col}_rolling_mean_20"] = data[col].rolling(20).mean()
        data[f"{col}_rolling_std_20"] = data[col].rolling(20).std()

    return data


def prepare_xy(data):
    """Return numeric features and the new future-price regression target."""
    target = "future_trade_price_5s"
    drop_cols = {
        target, "timestamp", "_1s_side", "_3s_side", "_5s_side"
    }

    x = data.drop(columns=[c for c in drop_cols if c in data.columns], errors="ignore")
    x = x.select_dtypes(include=[np.number])
    y = data[target]

    valid = y.notna() & x.notna().all(axis=1)
    return x.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def train_model(data, test_size=0.20, n_features=25):
    data = preprocessing(data)
    data = fill_null(data)
    data = create_regression_target(data, horizon_seconds=5)
    data = engineer_features(data)

    x, y = prepare_xy(data)

    split = int(len(x) * (1 - test_size))
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Assignment modification: Mutual Information + SelectKBest
    k = min(n_features, x_train.shape[1])
    selector = SelectKBest(score_func=mutual_info_regression, k=k)
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_test_selected = selector.transform(x_test)

    selected_features = x_train.columns[selector.get_support()].tolist()
    with open("features.txt", "w", encoding="utf-8") as f:
        json.dump({"keep_features": selected_features}, f, indent=2)

    # Assignment modification: two regression algorithms
    rf = RandomForestRegressor(
        n_estimators=150, max_depth=12, random_state=42,
        n_jobs=-1
    )
    hgb = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=0.1, random_state=42
    )

    rf.fit(x_train_selected, y_train)
    hgb.fit(x_train_selected, y_train)

    rf_pred = rf.predict(x_test_selected)
    hgb_pred = hgb.predict(x_test_selected)
    ensemble_pred = (rf_pred + hgb_pred) / 2

    metrics = {
        "MAE": float(mean_absolute_error(y_test, ensemble_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, ensemble_pred))),
        "R2": float(r2_score(y_test, ensemble_pred)),
        "test_rows": int(len(y_test)),
        "selected_features": selected_features,
    }

    joblib.dump(rf, os.path.join(PROJECT_DIR, "random_forest_regressor.joblib"))
    joblib.dump(hgb, os.path.join(PROJECT_DIR, "hist_gradient_boosting_regressor.joblib"))
    joblib.dump(selector, os.path.join(PROJECT_DIR, "feature_selector.joblib"))

    with open(os.path.join(PROJECT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Modified HFT Regression Results ===")
    print(f"Target: future trade price after 5 seconds")
    print(f"Training rows: {len(y_train)}")
    print(f"Testing rows: {len(y_test)}")
    print(f"Selected features: {len(selected_features)}")
    print(f"MAE:  {metrics['MAE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"R²:   {metrics['R2']:.6f}")

    return metrics


if __name__ == "__main__":
    # Always read/write project files relative to this script, not the terminal folder.
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(PROJECT_DIR, "data.csv")
    data = pd.read_csv(data_path)
    train_model(data)
