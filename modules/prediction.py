import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


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

    # Make sure data is chronologically ordered
    data = data.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    # Create previous-month features
    data["Previous_Temperature"] = (
        data["Temperature"].shift(1)
    )

    data["Previous_Minimum_Temperature"] = (
        data["Minimum_Temperature"].shift(1)
    )

    data["Previous_Maximum_Temperature"] = (
        data["Maximum_Temperature"].shift(1)
    )

    data["Previous_Rainfall"] = (
        data["Rainfall"].shift(1)
    )

    data["Previous_Pressure"] = (
        data["Pressure"].shift(1)
    )

    data["Previous_Sunshine"] = (
        data["Sunshine"].shift(1)
    )

    # Remove the first row because it has no
    # previous-month information
    data = data.dropna(
        subset=FEATURES + [TARGET]
    ).reset_index(
        drop=True
    )

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

    split_index = int(
        len(data) * 0.80
    )

    train_data = data.iloc[
        :split_index
    ]

    test_data = data.iloc[
        split_index:
    ]

    X_train = train_data[
        FEATURES
    ]

    y_train = train_data[
        TARGET
    ]

    X_test = test_data[
        FEATURES
    ]

    y_test = test_data[
        TARGET
    ]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# LINEAR REGRESSION
# ============================================================

def train_temperature_model(df):

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(df)

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
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

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(df)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=8
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return (
        model,
        mae,
        r2,
        y_test,
        predictions
    )


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
    )

    importance_data = importance_data.reset_index(
        drop=True
    )

    return importance_data