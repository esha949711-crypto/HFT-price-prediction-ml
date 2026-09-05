import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PROJECT DIRECTORY
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# NUMERIC COLUMNS
# ============================================================

NUMERIC_COLUMNS = [
    "bid_price",
    "bid_qty",
    "ask_price",
    "ask_qty",
    "trade_price",
    "sum_trade_1s",
    "bid_advance_time",
    "ask_advance_time",
    "last_trade_time"
]


# ============================================================
# 1. PREPROCESSING
# ============================================================

def preprocessing(data):
    """
    Convert data types and sort HFT observations chronologically.
    """

    data = data.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

    data = data.dropna(subset=["timestamp"])

    return data.sort_values(
        "timestamp"
    ).reset_index(drop=True)


# ============================================================
# 2. HANDLE MISSING VALUES
# ============================================================

def fill_null(data):
    """
    Fill missing trading activity values.
    """

    data = data.copy()

    if "sum_trade_1s" in data.columns:
        data["sum_trade_1s"] = (
            data["sum_trade_1s"]
            .fillna(0)
        )

    # Forward-fill market variables.
    market_columns = [
        "bid_price",
        "ask_price",
        "bid_qty",
        "ask_qty",
        "trade_price"
    ]

    for col in market_columns:
        if col in data.columns:
            data[col] = data[col].ffill()

    # Fill remaining numeric columns with median.
    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = data[col].fillna(
                data[col].median()
            )

    return data


# ============================================================
# 3. CREATE REGRESSION TARGET
# ============================================================

def create_regression_target(data, horizon_seconds=5):
    """
    Predict the first trade price observed at least
    5 seconds after the current timestamp.
    """

    data = data.copy()

    timestamps = (
        data["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    prices = (
        data["trade_price"]
        .to_numpy()
    )

    future_ns = (
        timestamps
        + int(horizon_seconds * 1e9)
    )

    future_index = np.searchsorted(
        timestamps,
        future_ns,
        side="left"
    )

    target = np.full(
        len(data),
        np.nan
    )

    valid = (
        future_index
        < len(data)
    )

    target[valid] = (
        prices[future_index[valid]]
    )

    data["future_trade_price_5s"] = target

    return data


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def engineer_features(data):
    """
    Build HFT market microstructure,
    lag, rolling and time-based features.
    """

    data = data.copy()

    # --------------------------------------------------------
    # Basic market features
    # --------------------------------------------------------

    data["spread"] = (
        data["ask_price"]
        - data["bid_price"]
    )

    data["mid_price"] = (
        data["ask_price"]
        + data["bid_price"]
    ) / 2

    data["price_to_mid"] = (
        data["trade_price"]
        - data["mid_price"]
    )

    data["bid_ask_qty_total"] = (
        data["bid_qty"]
        + data["ask_qty"]
    )

    data["bid_ask_qty_diff"] = (
        data["bid_qty"]
        - data["ask_qty"]
    )

    data["order_imbalance"] = (
        (data["bid_qty"] - data["ask_qty"])
        /
        (
            data["bid_qty"]
            + data["ask_qty"]
            + 1e-9
        )
    )

    # --------------------------------------------------------
    # Relative position inside bid/ask spread
    # --------------------------------------------------------

    data["trade_position"] = (
        (data["trade_price"] - data["bid_price"])
        /
        (
            data["ask_price"]
            - data["bid_price"]
            + 1e-9
        )
    )

    # --------------------------------------------------------
    # Price movement features
    # --------------------------------------------------------

    data["trade_price_change"] = (
        data["trade_price"].diff()
    )

    data["mid_price_change"] = (
        data["mid_price"].diff()
    )

    data["bid_price_change"] = (
        data["bid_price"].diff()
    )

    data["ask_price_change"] = (
        data["ask_price"].diff()
    )

    # --------------------------------------------------------
    # Trade vs bid/ask
    # --------------------------------------------------------

    data["trade_price_compare"] = 0

    data.loc[
        data["trade_price"] <= data["bid_price"],
        "trade_price_compare"
    ] = -1

    data.loc[
        data["trade_price"] >= data["ask_price"],
        "trade_price_compare"
    ] = 1

    # --------------------------------------------------------
    # Base columns for lag features
    # --------------------------------------------------------

    base_cols = [
        "bid_price",
        "bid_qty",
        "ask_price",
        "ask_qty",
        "trade_price",
        "sum_trade_1s",
        "bid_advance_time",
        "ask_advance_time",
        "last_trade_time",
        "spread",
        "mid_price",
        "price_to_mid",
        "bid_ask_qty_total",
        "bid_ask_qty_diff",
        "order_imbalance",
        "trade_position",
        "trade_price_compare"
    ]

    # --------------------------------------------------------
    # Lag and difference features
    # --------------------------------------------------------

    for col in base_cols:

        data[f"{col}_diff"] = (
            data[col].diff()
        )

        data[f"{col}_lag_1"] = (
            data[col].shift(1)
        )

        data[f"{col}_lag_2"] = (
            data[col].shift(2)
        )

        data[f"{col}_lag_5"] = (
            data[col].shift(5)
        )

    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    rolling_cols = [
        "trade_price",
        "spread",
        "mid_price",
        "bid_qty",
        "ask_qty",
        "sum_trade_1s",
        "order_imbalance"
    ]

    for col in rolling_cols:

        data[f"{col}_rolling_mean_5"] = (
            data[col]
            .rolling(5)
            .mean()
        )

        data[f"{col}_rolling_mean_20"] = (
            data[col]
            .rolling(20)
            .mean()
        )

        data[f"{col}_rolling_std_20"] = (
            data[col]
            .rolling(20)
            .std()
        )

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    data["hour"] = (
        data["timestamp"].dt.hour
    )

    data["minute"] = (
        data["timestamp"].dt.minute
    )

    data["second"] = (
        data["timestamp"].dt.second
    )

    data["millisecond"] = (
        data["timestamp"].dt.microsecond / 1000
    )

    # --------------------------------------------------------
    # Cyclic time representation
    # --------------------------------------------------------

    seconds_of_day = (
        data["hour"] * 3600
        + data["minute"] * 60
        + data["second"]
    )

    data["time_sin"] = np.sin(
        2 * np.pi * seconds_of_day / 86400
    )

    data["time_cos"] = np.cos(
        2 * np.pi * seconds_of_day / 86400
    )

    return data


# ============================================================
# 5. PREPARE X AND Y
# ============================================================

def prepare_xy(data):

    target = "future_trade_price_5s"

    drop_cols = {
        target,
        "timestamp",
        "_1s_side",
        "_3s_side",
        "_5s_side"
    }

    x = data.drop(
        columns=[
            c for c in drop_cols
            if c in data.columns
        ],
        errors="ignore"
    )

    x = x.select_dtypes(
        include=[np.number]
    )

    y = data[target]

    valid = (
        y.notna()
        &
        x.notna().all(axis=1)
    )

    x = x.loc[valid].reset_index(
        drop=True
    )

    y = y.loc[valid].reset_index(
        drop=True
    )

    return x, y


# ============================================================
# 6. TRAIN MODEL
# ============================================================

def train_model(
    data,
    test_size=0.20,
    n_features=25
):

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    data = preprocessing(data)

    data = fill_null(data)

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    data = create_regression_target(
        data,
        horizon_seconds=5
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    data = engineer_features(data)

    # --------------------------------------------------------
    # X AND Y
    # --------------------------------------------------------

    x, y = prepare_xy(data)

    print()
    print("Total usable rows:", len(x))
    print("Total features before selection:", x.shape[1])

    # --------------------------------------------------------
    # TIME-BASED TRAIN/TEST SPLIT
    # --------------------------------------------------------

    split = int(
        len(x) * (1 - test_size)
    )

    x_train = x.iloc[:split]
    x_test = x.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # --------------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------------

    k = min(
        n_features,
        x_train.shape[1]
    )

    selector = SelectKBest(
        score_func=mutual_info_regression,
        k=k
    )

    x_train_selected = (
        selector.fit_transform(
            x_train,
            y_train
        )
    )

    x_test_selected = (
        selector.transform(
            x_test
        )
    )

    selected_features = (
        x_train
        .columns[
            selector.get_support()
        ]
        .tolist()
    )

    # Save selected features
    with open(
        os.path.join(
            PROJECT_DIR,
            "features.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "keep_features":
                selected_features
            },
            f,
            indent=2
        )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf = RandomForestRegressor(
        n_estimators=250,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    # --------------------------------------------------------
    # HISTOGRAM GRADIENT BOOSTING
    # --------------------------------------------------------

    hgb = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print("Training Random Forest...")

    rf.fit(
        x_train_selected,
        y_train
    )

    print("Training HistGradientBoosting...")

    hgb.fit(
        x_train_selected,
        y_train
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    rf_pred = rf.predict(
        x_test_selected
    )

    hgb_pred = hgb.predict(
        x_test_selected
    )

    # Average ensemble
    ensemble_pred = (
        rf_pred + hgb_pred
    ) / 2

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    # A simple baseline:
    # predict the current trade price as future price.

    baseline_pred = (
        x_test["trade_price"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    model_mae = mean_absolute_error(
        y_test,
        ensemble_pred
    )

    model_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            ensemble_pred
        )
    )

    model_r2 = r2_score(
        y_test,
        ensemble_pred
    )

    # --------------------------------------------------------
    # BASELINE METRICS
    # --------------------------------------------------------

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_pred
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_pred
        )
    )

    baseline_r2 = r2_score(
        y_test,
        baseline_pred
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    metrics = {

        "MAE": float(model_mae),

        "RMSE": float(model_rmse),

        "R2": float(model_r2),

        "baseline_MAE": float(
            baseline_mae
        ),

        "baseline_RMSE": float(
            baseline_rmse
        ),

        "baseline_R2": float(
            baseline_r2
        ),

        "training_rows": int(
            len(y_train)
        ),

        "test_rows": int(
            len(y_test)
        ),

        "selected_features":
        selected_features
    }

    # --------------------------------------------------------
    # SAVE MODELS
    # --------------------------------------------------------

    joblib.dump(
        rf,
        os.path.join(
            PROJECT_DIR,
            "random_forest_regressor.joblib"
        )
    )

    joblib.dump(
        hgb,
        os.path.join(
            PROJECT_DIR,
            "hist_gradient_boosting_regressor.joblib"
        )
    )

    joblib.dump(
        selector,
        os.path.join(
            PROJECT_DIR,
            "feature_selector.joblib"
        )
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    with open(
        os.path.join(
            PROJECT_DIR,
            "metrics.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # PRINT FINAL RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("HFT FUTURE TRADE PRICE REGRESSION RESULTS")
    print("=" * 60)

    print(
        f"Target: Future trade price after 5 seconds"
    )

    print(
        f"Training rows: {len(y_train)}"
    )

    print(
        f"Testing rows: {len(y_test)}"
    )

    print(
        f"Selected features: {len(selected_features)}"
    )

    print()
    print("----- MACHINE LEARNING ENSEMBLE -----")

    print(
        f"MAE:  {model_mae:.6f}"
    )

    print(
        f"RMSE: {model_rmse:.6f}"
    )

    print(
        f"R²:   {model_r2:.6f}"
    )

    print()
    print("----- NAIVE BASELINE -----")

    print(
        f"MAE:  {baseline_mae:.6f}"
    )

    print(
        f"RMSE: {baseline_rmse:.6f}"
    )

    print(
        f"R²:   {baseline_r2:.6f}"
    )

    print()
    print("----- COMPARISON -----")

    if model_mae < baseline_mae:
        print(
            "MAE: Model is better than baseline."
        )
    else:
        print(
            "MAE: Model is NOT better than baseline."
        )

    if model_rmse < baseline_rmse:
        print(
            "RMSE: Model is better than baseline."
        )
    else:
        print(
            "RMSE: Model is NOT better than baseline."
        )

    if model_r2 > baseline_r2:
        print(
            "R²: Model is better than baseline."
        )
    else:
        print(
            "R²: Model is NOT better than baseline."
        )

    print("=" * 60)

    return metrics


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    data_path = os.path.join(
        PROJECT_DIR,
        "data.csv"
    )

    if not os.path.exists(data_path):

        print(
            "ERROR: data.csv was not found."
        )

        print(
            "Expected location:"
        )

        print(data_path)

        raise FileNotFoundError(
            data_path
        )

    print(
        "Loading dataset..."
    )

    data = pd.read_csv(
        data_path
    )

    print(
        "Dataset loaded successfully."
    )

    train_model(data)