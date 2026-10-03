import pandas as pd

from sklearn.ensemble import IsolationForest


# ============================================================
# REAL CLIMATE FEATURES
# ============================================================

FEATURES = [
    "Temperature",
    "Minimum_Temperature",
    "Maximum_Temperature",
    "Rainfall",
    "Pressure",
    "Sunshine"
]


# ============================================================
# ANOMALY DETECTION
# ============================================================

def detect_anomalies(df):

    # Check required columns
    for column in FEATURES:

        if column not in df.columns:

            raise ValueError(
                f"Required column '{column}' is missing."
            )

    # Select real climate variables
    X = df[FEATURES].copy()

    # Create Isolation Forest
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    # Detect anomalies
    predictions = model.fit_predict(X)

    # Create result dataframe
    result = df.copy()

    # -1 = anomaly
    #  1 = normal
    result["Anomaly"] = predictions

    result["Anomaly_Status"] = result[
        "Anomaly"
    ].map({
        1: "Normal",
        -1: "Anomaly"
    })

    return result