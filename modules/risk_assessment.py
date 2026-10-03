import pandas as pd


# ============================================================
# PROJECT-DEFINED CLIMATE RISK MODEL
# ============================================================

def calculate_risk_score(df):

    data = df.copy()

    required_columns = [
        "Temperature",
        "Maximum_Temperature",
        "Rainfall",
        "Pressure"
    ]

    for column in required_columns:

        if column not in data.columns:

            raise ValueError(
                f"Required column '{column}' is missing."
            )

    # --------------------------------------------------------
    # TEMPERATURE RISK
    # Higher temperature = higher risk
    # --------------------------------------------------------

    temp_min = data["Temperature"].min()
    temp_max = data["Temperature"].max()

    if temp_max == temp_min:

        data["Temperature_Risk"] = 0

    else:

        data["Temperature_Risk"] = (
            (
                data["Temperature"] - temp_min
            )
            /
            (temp_max - temp_min)
        ) * 100


    # --------------------------------------------------------
    # MAXIMUM TEMPERATURE RISK
    # Higher maximum temperature = higher risk
    # --------------------------------------------------------

    max_temp_min = data[
        "Maximum_Temperature"
    ].min()

    max_temp_max = data[
        "Maximum_Temperature"
    ].max()

    if max_temp_max == max_temp_min:

        data["Maximum_Temperature_Risk"] = 0

    else:

        data["Maximum_Temperature_Risk"] = (
            (
                data["Maximum_Temperature"]
                - max_temp_min
            )
            /
            (max_temp_max - max_temp_min)
        ) * 100


    # --------------------------------------------------------
    # RAINFALL RISK
    # Higher rainfall = higher project-defined risk
    # --------------------------------------------------------

    rainfall_min = data["Rainfall"].min()
    rainfall_max = data["Rainfall"].max()

    if rainfall_max == rainfall_min:

        data["Rainfall_Risk"] = 0

    else:

        data["Rainfall_Risk"] = (
            (
                data["Rainfall"] - rainfall_min
            )
            /
            (rainfall_max - rainfall_min)
        ) * 100


    # --------------------------------------------------------
    # PRESSURE RISK
    # Lower pressure = higher project-defined risk
    # --------------------------------------------------------

    pressure_min = data["Pressure"].min()
    pressure_max = data["Pressure"].max()

    if pressure_max == pressure_min:

        data["Pressure_Risk"] = 0

    else:

        data["Pressure_Risk"] = (
            (
                pressure_max - data["Pressure"]
            )
            /
            (pressure_max - pressure_min)
        ) * 100


    # --------------------------------------------------------
    # COMBINED CLIMATE RISK SCORE
    #
    # These weights are project-defined:
    #
    # Temperature           = 30%
    # Maximum Temperature   = 30%
    # Rainfall              = 25%
    # Pressure              = 15%
    # --------------------------------------------------------

    data["Climate_Risk_Score"] = (
        data["Temperature_Risk"] * 0.30
        + data["Maximum_Temperature_Risk"] * 0.30
        + data["Rainfall_Risk"] * 0.25
        + data["Pressure_Risk"] * 0.15
    )


    # --------------------------------------------------------
    # RISK CATEGORY
    # --------------------------------------------------------

    def classify_risk(score):

        if score < 25:

            return "Low"

        elif score < 50:

            return "Moderate"

        elif score < 75:

            return "High"

        else:

            return "Critical"


    data["Risk_Category"] = (
        data["Climate_Risk_Score"]
        .apply(classify_risk)
    )


    return data