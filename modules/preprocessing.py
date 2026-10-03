import pandas as pd


def preprocess_data(df):
    """
    Clean and prepare the real Hyderabad climate dataset.
    """

    data = df.copy()

    # --------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------

    data = data.drop_duplicates()


    # --------------------------------------------------
    # CONVERT DATE
    # --------------------------------------------------

    if "Date" in data.columns:

        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )


    # --------------------------------------------------
    # REAL CLIMATE NUMERIC COLUMNS
    # --------------------------------------------------

    numeric_columns = [
        "Temperature",
        "Minimum_Temperature",
        "Maximum_Temperature",
        "Rainfall",
        "Pressure",
        "Sunshine"
    ]


    # --------------------------------------------------
    # CONVERT NUMERIC VALUES
    # --------------------------------------------------

    for column in numeric_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )


    # --------------------------------------------------
    # HANDLE MISSING VALUES
    # --------------------------------------------------

    for column in numeric_columns:

        if column in data.columns:

            median_value = data[column].median()

            data[column] = data[column].fillna(
                median_value
            )


    # --------------------------------------------------
    # SORT BY DATE
    # --------------------------------------------------

    if "Date" in data.columns:

        data = data.sort_values(
            by="Date"
        )


    # --------------------------------------------------
    # RESET INDEX
    # --------------------------------------------------

    data = data.reset_index(
        drop=True
    )


    return data