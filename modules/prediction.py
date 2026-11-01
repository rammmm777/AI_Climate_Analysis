
import os
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# MODEL SAVE DIRECTORY
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(PROJECT_DIR, "saved_models")

os.makedirs(MODEL_DIR, exist_ok=True)

LINEAR_MODEL_PATH = os.path.join(
    MODEL_DIR, "linear_regression.pkl"
)

RANDOM_FOREST_MODEL_PATH = os.path.join(
    MODEL_DIR, "random_forest.pkl"
)


# ============================================================
# FORECAST FEATURES
# ============================================================

FEATURES = [
    "Previous_Temperature",
    "Previous_Minimum_Temperature",
    "Previous_Maximum_Temperature",
    "Previous_Rainfall",
    "Previous_Pressure",
    "Previous_Sunshine"
]

TARGET = "Temperature"


# ============================================================
# PREPARE TIME-SERIES DATA
# ============================================================

def prepare_data(df):

    required_columns = [
        "Date",
        "Temperature",
        "Minimum_Temperature",
        "Maximum_Temperature",
        "Rainfall",
        "Pressure",
        "Sunshine"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' is missing."
            )

    data = df.copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Date"]
    )

    data = data.sort_values(
        "Date"
    ).reset_index(drop=True)

    previous_features = {
        "Previous_Temperature": "Temperature",
        "Previous_Minimum_Temperature": "Minimum_Temperature",
        "Previous_Maximum_Temperature": "Maximum_Temperature",
        "Previous_Rainfall": "Rainfall",
        "Previous_Pressure": "Pressure",
        "Previous_Sunshine": "Sunshine"
    }

    for feature, source_column in previous_features.items():
        data[feature] = data[source_column].shift(1)

    data = data.dropna(
        subset=FEATURES + [TARGET]
    ).reset_index(drop=True)

    return data


# ============================================================
# TIME-ORDERED TRAIN / TEST SPLIT
# ============================================================

def split_data(df):

    data = prepare_data(df)

    if len(data) < 10:
        raise ValueError(
            "Not enough records available for time-series prediction."
        )

    split_index = int(len(data) * 0.80)

    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]

    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    return X_train, X_test, y_train, y_test


# ============================================================
# LINEAR REGRESSION
# ============================================================

def train_temperature_model(df):

    X_train, X_test, y_train, y_test = split_data(df)

    model = LinearRegression()

    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, LINEAR_MODEL_PATH)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test, predictions
    )

    r2 = r2_score(
        y_test, predictions
    )

    return (
        model,
        mae,
        r2,
        y_test,
        predictions
    )


# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest_model(df):

    X_train, X_test, y_train, y_test = split_data(df)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=8
    )

    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, RANDOM_FOREST_MODEL_PATH)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test, predictions
    )

    r2 = r2_score(
        y_test, predictions
    )

    return (
        model,
        mae,
        r2,
        y_test,
        predictions
    )


# ============================================================
# LOAD SAVED MODELS
# ============================================================

def load_saved_models():

    linear_model = None
    random_forest_model = None

    if os.path.exists(LINEAR_MODEL_PATH):
        linear_model = joblib.load(LINEAR_MODEL_PATH)

    if os.path.exists(RANDOM_FOREST_MODEL_PATH):
        random_forest_model = joblib.load(
            RANDOM_FOREST_MODEL_PATH
        )

    return linear_model, random_forest_model


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(model):

    importance_data = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    })

    importance_data = importance_data.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    return importance_data
