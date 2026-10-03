from pathlib import Path


# ============================================================
# PROJECT BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODULES_DIR = BASE_DIR / "modules"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"


# ============================================================
# MAIN DATASETS
# ============================================================

CLIMATE_DATA_FILE = DATA_DIR / "hyderabad_climate_2000_2025.csv"

INTEGRATED_ENVIRONMENT_FILE = (
    DATA_DIR / "integrated_climate_environment_2000_2024.csv"
)

INDIA_CO2_FILE = DATA_DIR / "india_co2_2000_2024.csv"

INDIA_ENERGY_FILE = DATA_DIR / "india_energy_2000_2024.csv"


# ============================================================
# DATABASE
# ============================================================

DATABASE_FILE = OUTPUTS_DIR / "climate_analysis.db"


# ============================================================
# GENERATED REPORT
# ============================================================

FINAL_REPORT_FILE = OUTPUTS_DIR / "AI_Climate_Analysis_Final_Report.html"


# ============================================================
# CREATE REQUIRED DIRECTORIES IF THEY DO NOT EXIST
# ============================================================

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
MODULES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)