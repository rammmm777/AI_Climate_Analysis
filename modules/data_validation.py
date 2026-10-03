REQUIRED_CLIMATE_COLUMNS = [
    "Date",
    "Temperature",
    "Minimum_Temperature",
    "Maximum_Temperature",
    "Rainfall",
    "Pressure",
    "Sunshine"
]


def validate_climate_data(df):

    missing_columns = [
        column
        for column in REQUIRED_CLIMATE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        return False, missing_columns

    return True, []