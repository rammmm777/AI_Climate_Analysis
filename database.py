import sqlite3
import pandas as pd
from pathlib import Path


# ============================================================
# DATABASE LOCATION
# ============================================================

DATABASE_PATH = Path(
    "outputs/climate_analysis.db"
)


# ============================================================
# CREATE DATABASE
# ============================================================

def create_database():

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS climate_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            temperature REAL,
            minimum_temperature REAL,
            maximum_temperature REAL,
            rainfall REAL,
            pressure REAL,
            sunshine REAL
        )
    """)

    connection.commit()

    connection.close()


# ============================================================
# SAVE CLIMATE DATA
# ============================================================

def save_climate_data(df):

    create_database()

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    data = df.copy()

    # Convert date to text for SQLite
    data["Date"] = data["Date"].astype(str)

    data = data.rename(
        columns={
            "Date": "date",
            "Temperature": "temperature",
            "Minimum_Temperature": "minimum_temperature",
            "Maximum_Temperature": "maximum_temperature",
            "Rainfall": "rainfall",
            "Pressure": "pressure",
            "Sunshine": "sunshine"
        }
    )

    required_columns = [
        "date",
        "temperature",
        "minimum_temperature",
        "maximum_temperature",
        "rainfall",
        "pressure",
        "sunshine"
    ]

    data = data[
        required_columns
    ]

    data.to_sql(
        "climate_data",
        connection,
        if_exists="replace",
        index=False
    )

    connection.close()


# ============================================================
# LOAD SAVED DATA
# ============================================================

def load_saved_data():

    create_database()

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    data = pd.read_sql_query(
        "SELECT * FROM climate_data",
        connection
    )

    connection.close()

    return data