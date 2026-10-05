import streamlit as st
import pandas as pd

from config import (
    CLIMATE_DATA_FILE,
    INTEGRATED_ENVIRONMENT_FILE,
    FINAL_REPORT_FILE,
)

from modules.preprocessing import preprocess_data
from modules.data_validation import validate_climate_data

from modules.prediction import (
    train_temperature_model,
    train_random_forest_model,
    get_feature_importance,
)

from modules.anomaly_detection import detect_anomalies
from modules.risk_assessment import calculate_risk_score

from database import save_climate_data

from modules.pdf_analysis import extract_text_from_pdf
from modules.esg_analysis import analyze_esg_text
from modules.report_generation import generate_analysis_report


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Powered Climate Analysis",
    page_icon="🌍",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "esg_result" not in st.session_state:
    st.session_state.esg_result = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_first_existing_column(df, candidates):
    """Return the first matching column from a list."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def safe_numeric(series):
    """Convert a Series to numeric values."""
    return pd.to_numeric(series, errors="coerce")


def load_environmental_data():
    """Load the integrated annual environmental dataset."""
    if not INTEGRATED_ENVIRONMENT_FILE.exists():
        return None

    env_df = pd.read_csv(
        INTEGRATED_ENVIRONMENT_FILE
    )

    if "year" in env_df.columns:
        env_df["year"] = pd.to_numeric(
            env_df["year"],
            errors="coerce"
        )

    env_df = env_df.dropna(
        subset=["year"]
    ).copy()

    env_df["year"] = env_df["year"].astype(int)

    return env_df


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🌍 Climate Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "EDA",
        "AI Prediction",
        "Anomaly Detection",
        "Risk Assessment",
        "Environmental Analysis",
        "ESG Analysis",
        "Report",
    ],
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🌍 AI-Powered Climate Analysis System"
)

st.write(
    "An AI-based system for climate analysis, prediction, "
    "anomaly detection, environmental analysis, ESG analysis, "
    "and climate-risk assessment."
)


# ============================================================
# CLIMATE DATASET
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "📂 Climate Dataset"
)

uploaded_climate_file = st.sidebar.file_uploader(
    "Upload a climate CSV file",
    type=["csv"],
)


# ============================================================
# LOAD CLIMATE DATA
# ============================================================

df = None

if uploaded_climate_file is not None:

    try:

        uploaded_df = pd.read_csv(
            uploaded_climate_file
        )

        is_valid, missing_columns = validate_climate_data(
            uploaded_df
        )

        if is_valid:

            df = uploaded_df

            st.sidebar.success(
                "Uploaded climate CSV loaded successfully."
            )

        else:

            st.sidebar.error(
                "Invalid climate dataset."
            )

            st.sidebar.warning(
                "Missing columns: "
                + ", ".join(missing_columns)
            )

    except Exception as error:

        st.sidebar.error(
            f"Unable to read uploaded CSV: {error}"
        )


elif CLIMATE_DATA_FILE.exists():

    try:

        df = pd.read_csv(
            CLIMATE_DATA_FILE
        )

        st.sidebar.info(
            "Using the main Hyderabad climate dataset (2000–2025)."
        )

    except Exception as error:

        st.sidebar.error(
            f"Unable to read default climate dataset: {error}"
        )


else:

    st.sidebar.error(
        "Main climate dataset was not found."
    )

    st.sidebar.caption(
        f"Expected file: {CLIMATE_DATA_FILE}"
    )


# ============================================================
# PREPROCESS DATA
# ============================================================

if df is not None:

    try:

        df = preprocess_data(df)

    except Exception as error:

        st.error(
            f"Data preprocessing failed: {error}"
        )

        df = None


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    if df is None:

        st.warning(
            "Please provide a valid climate dataset."
        )

    else:

        st.header(
            "📊 Dataset Overview"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Records",
                df.shape[0]
            )

        with col2:
            st.metric(
                "Total Features",
                df.shape[1]
            )

        with col3:
            st.metric(
                "Missing Values",
                int(
                    df.isnull()
                    .sum()
                    .sum()
                )
            )

        st.success(
            "Climate dataset loaded and preprocessed successfully."
        )

        st.subheader(
            "📋 Climate Data"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        st.subheader(
            "ℹ️ Dataset Information"
        )

        if "Date" in df.columns:

            st.write(
                f"Date range: "
                f"{df['Date'].min().date()} "
                f"to "
                f"{df['Date'].max().date()}"
            )

        st.write(
            "The main climate dataset contains monthly "
            "weather observations for Hyderabad."
        )


# ============================================================
# EDA
# ============================================================

elif page == "EDA":

    if df is None:

        st.warning(
            "Please provide a valid climate dataset."
        )

    else:

        st.header(
            "📈 Exploratory Data Analysis"
        )

        st.write(
            "Explore historical climate patterns using "
            "temperature, rainfall, pressure, and sunshine data."
        )

        # ----------------------------------------------------
        # TEMPERATURE TREND
        # ----------------------------------------------------

        st.subheader(
            "🌡️ Temperature Trend"
        )

        if {
            "Date",
            "Temperature"
        }.issubset(df.columns):

            temperature_data = (
                df.set_index("Date")
                [["Temperature"]]
            )

            st.line_chart(
                temperature_data
            )

        # ----------------------------------------------------
        # MINIMUM VS MAXIMUM TEMPERATURE
        # ----------------------------------------------------

        if {
            "Date",
            "Minimum_Temperature",
            "Maximum_Temperature"
        }.issubset(df.columns):

            st.subheader(
                "🌡️ Minimum vs Maximum Temperature"
            )

            temperature_range = (
                df.set_index("Date")
                [
                    [
                        "Minimum_Temperature",
                        "Maximum_Temperature"
                    ]
                ]
            )

            st.line_chart(
                temperature_range
            )

        # ----------------------------------------------------
        # RAINFALL
        # ----------------------------------------------------

        st.subheader(
            "🌧️ Rainfall Trend"
        )

        if {
            "Date",
            "Rainfall"
        }.issubset(df.columns):

            rainfall_data = (
                df.set_index("Date")
                [["Rainfall"]]
            )

            st.line_chart(
                rainfall_data
            )

        # ----------------------------------------------------
        # PRESSURE
        # ----------------------------------------------------

        st.subheader(
            "🌬️ Atmospheric Pressure Trend"
        )

        if {
            "Date",
            "Pressure"
        }.issubset(df.columns):

            pressure_data = (
                df.set_index("Date")
                [["Pressure"]]
            )

            st.line_chart(
                pressure_data
            )

        # ----------------------------------------------------
        # SUNSHINE
        # ----------------------------------------------------

        st.subheader(
            "☀️ Sunshine Trend"
        )

        if {
            "Date",
            "Sunshine"
        }.issubset(df.columns):

            sunshine_data = (
                df.set_index("Date")
                [["Sunshine"]]
            )

            st.line_chart(
                sunshine_data
            )

        # ----------------------------------------------------
        # STATISTICAL SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "📊 Statistical Summary"
        )

        st.dataframe(
            df.describe(),
            use_container_width=True
        )

        # ----------------------------------------------------
        # CORRELATION ANALYSIS
        # ----------------------------------------------------

        st.subheader(
            "🔗 Climate Variable Correlation"
        )

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

        if len(numeric_columns) >= 2:

            correlation_matrix = (
                df[numeric_columns]
                .corr()
                .round(2)
            )

            st.dataframe(
                correlation_matrix,
                use_container_width=True
            )

            st.caption(
                "Correlation values range from -1 to +1. "
                "Positive values indicate that variables tend "
                "to increase together, while negative values "
                "indicate an inverse relationship. Correlation "
                "does not establish causation."
            )

        else:

            st.info(
                "Not enough numerical variables are available "
                "for correlation analysis."
            )


# ============================================================
# AI PREDICTION
# ============================================================

elif page == "AI Prediction":

    if df is None:

        st.warning(
            "Please provide a valid climate dataset."
        )

    else:

        st.header(
            "🤖 One-Month-Ahead Temperature Prediction"
        )

        st.write(
            "The system predicts the next month's average "
            "temperature using climate conditions from the "
            "previous month."
        )

        try:

            # ------------------------------------------------
            # LINEAR REGRESSION
            # ------------------------------------------------

            (
                linear_model,
                linear_mae,
                linear_r2,
                y_test,
                linear_predictions,
            ) = train_temperature_model(
                df
            )

            # ------------------------------------------------
            # RANDOM FOREST
            # ------------------------------------------------

            (
                rf_model,
                rf_mae,
                rf_r2,
                _,
                rf_predictions,
            ) = train_random_forest_model(
                df
            )

            # ------------------------------------------------
            # MODEL PERFORMANCE
            # ------------------------------------------------

            st.subheader(
                "📊 Model Performance"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "### Linear Regression"
                )

                st.metric(
                    "Mean Absolute Error",
                    f"{linear_mae:.2f} °C"
                )

                st.metric(
                    "R² Score",
                    f"{linear_r2:.2f}"
                )

            with col2:

                st.markdown(
                    "### Random Forest"
                )

                st.metric(
                    "Mean Absolute Error",
                    f"{rf_mae:.2f} °C"
                )

                st.metric(
                    "R² Score",
                    f"{rf_r2:.2f}"
                )

            # ------------------------------------------------
            # ACTUAL VS PREDICTED
            # ------------------------------------------------

            st.subheader(
                "📈 Actual vs Predicted Temperature"
            )

            prediction_data = pd.DataFrame(
                {
                    "Actual Temperature": y_test.values,
                    "Linear Regression": linear_predictions,
                    "Random Forest": rf_predictions,
                }
            )

            st.dataframe(
                prediction_data,
                use_container_width=True
            )

            # ------------------------------------------------
            # DOWNLOAD PREDICTIONS
            # ------------------------------------------------

            st.download_button(
                label="📥 Download Prediction Results",
                data=prediction_data
                .to_csv(index=False)
                .encode("utf-8"),
                file_name="temperature_prediction_results.csv",
                mime="text/csv",
            )

            # ------------------------------------------------
            # BETTER MODEL
            # ------------------------------------------------

            if rf_mae < linear_mae:

                st.success(
                    "Random Forest has the lower MAE "
                    "on the current test set."
                )

            else:

                st.success(
                    "Linear Regression has the lower MAE "
                    "on the current test set."
                )

            # ------------------------------------------------
            # MODEL COMPARISON
            # ------------------------------------------------

            st.subheader(
                "📊 Model Comparison"
            )

            model_comparison = pd.DataFrame(
                {
                    "Model": [
                        "Linear Regression",
                        "Random Forest"
                    ],
                    "MAE (°C)": [
                        round(linear_mae, 2),
                        round(rf_mae, 2)
                    ],
                    "R² Score": [
                        round(linear_r2, 2),
                        round(rf_r2, 2)
                    ]
                }
            )

            st.dataframe(
                model_comparison,
                use_container_width=True
            )

            # ------------------------------------------------
            # FEATURE IMPORTANCE
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🔎 Random Forest Feature Importance"
            )

            try:

                importance_df = get_feature_importance(
                    rf_model
                )

                if (
                    importance_df is not None
                    and not importance_df.empty
                ):

                    feature_column = (
                        importance_df.columns[0]
                    )

                    st.bar_chart(
                        importance_df.set_index(
                            feature_column
                        )
                    )

                    st.dataframe(
                        importance_df,
                        use_container_width=True
                    )

                st.caption(
                    "Feature importance represents the relative "
                    "contribution of previous-month climate "
                    "variables to the Random Forest prediction. "
                    "It does not establish causation."
                )

            except Exception as error:

                st.warning(
                    "Feature importance could not be displayed: "
                    f"{error}"
                )

        except ValueError as error:

            st.error(
                str(error)
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    if df is None:

        st.warning(
            "Please provide a valid climate dataset."
        )

    else:

        st.header(
            "🔍 Climate Anomaly Detection"
        )

        st.write(
            "Isolation Forest is used to identify climate "
            "observations that are statistically unusual."
        )

        try:

            anomaly_data = detect_anomalies(
                df
            )

            anomaly_count = int(
                (
                    anomaly_data["Anomaly_Status"]
                    == "Anomaly"
                ).sum()
            )

            normal_count = int(
                (
                    anomaly_data["Anomaly_Status"]
                    == "Normal"
                ).sum()
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Normal Records",
                    normal_count
                )

            with col2:

                st.metric(
                    "Anomalies Detected",
                    anomaly_count
                )

            st.subheader(
                "📋 Anomaly Detection Results"
            )

            st.dataframe(
                anomaly_data,
                use_container_width=True
            )

            # ------------------------------------------------
            # DOWNLOAD ANOMALIES
            # ------------------------------------------------

            st.download_button(
                label="📥 Download Anomaly Results",
                data=anomaly_data
                .to_csv(index=False)
                .encode("utf-8"),
                file_name="climate_anomaly_results.csv",
                mime="text/csv",
            )

            if anomaly_count > 0:

                st.warning(
                    f"The system detected {anomaly_count} "
                    "statistically unusual climate record(s)."
                )

            else:

                st.success(
                    "No unusual climate records were detected."
                )

            st.caption(
                "An anomaly is a statistical result from the "
                "machine-learning model and is not automatically "
                "a confirmed extreme climate event."
            )

        except ValueError as error:

            st.error(
                str(error)
            )

        except Exception as error:

            st.error(
                f"Anomaly detection failed: {error}"
            )


# ============================================================
# RISK ASSESSMENT
# ============================================================

elif page == "Risk Assessment":

    if df is None:

        st.warning(
            "Please provide a valid climate dataset."
        )

    else:

        st.header(
            "⚠️ Climate Risk Assessment"
        )

        st.write(
            "The system calculates a project-defined climate "
            "risk score using selected climate variables."
        )

        try:

            risk_data = calculate_risk_score(
                df
            )

            average_risk = (
                risk_data["Climate_Risk_Score"]
                .mean()
            )

            highest_risk = risk_data.loc[
                risk_data["Climate_Risk_Score"].idxmax(),
                "Risk_Category"
            ]

            risk_counts = (
                risk_data["Risk_Category"]
                .value_counts()
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Average Climate Risk Score",
                    f"{average_risk:.2f} / 100"
                )

            with col2:

                st.metric(
                    "Highest Risk Category",
                    highest_risk
                )

            # ------------------------------------------------
            # CATEGORY SUMMARY
            # ------------------------------------------------

            st.subheader(
                "📊 Risk Category Summary"
            )

            st.bar_chart(
                risk_counts
            )

            # ------------------------------------------------
            # RESULTS TABLE
            # ------------------------------------------------

            st.subheader(
                "📋 Climate Risk Results"
            )

            display_columns = [
                "Date",
                "Temperature",
                "Minimum_Temperature",
                "Maximum_Temperature",
                "Rainfall",
                "Pressure",
                "Climate_Risk_Score",
                "Risk_Category",
            ]

            available_columns = [
                column
                for column in display_columns
                if column in risk_data.columns
            ]

            risk_export = (
                risk_data[
                    available_columns
                ].copy()
            )

            st.dataframe(
                risk_export,
                use_container_width=True
            )

            # ------------------------------------------------
            # DOWNLOAD RISK RESULTS
            # ------------------------------------------------

            st.download_button(
                label="📥 Download Risk Results",
                data=risk_export
                .to_csv(index=False)
                .encode("utf-8"),
                file_name="climate_risk_results.csv",
                mime="text/csv",
            )

            # ------------------------------------------------
            # DATABASE
            # ------------------------------------------------

            st.subheader(
                "💾 Database"
            )

            if st.button(
                "Save Climate Data to Database"
            ):

                try:

                    save_climate_data(
                        df
                    )

                    st.success(
                        "Climate data saved successfully "
                        "to SQLite database."
                    )

                except Exception as error:

                    st.error(
                        f"Database save failed: {error}"
                    )

            st.caption(
                "Risk scores and categories are based on the "
                "project-defined risk methodology."
            )

        except ValueError as error:

            st.error(
                str(error)
            )

        except Exception as error:

            st.error(
                f"Risk assessment failed: {error}"
            )


# ============================================================
# ENVIRONMENTAL ANALYSIS
# ============================================================

elif page == "Environmental Analysis":

    st.header(
        "🌍 Environmental Analysis"
    )

    st.write(
        "Integrated annual climate, CO₂/GHG, and energy "
        "analysis for 2000–2024."
    )

    try:

        env_df = load_environmental_data()

        if env_df is None or env_df.empty:

            st.warning(
                "Integrated environmental dataset was not found."
            )

            st.info(
                f"Expected file: "
                f"{INTEGRATED_ENVIRONMENT_FILE}"
            )

        else:

            start_year = int(
                env_df["year"].min()
            )

            end_year = int(
                env_df["year"].max()
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Years",
                    len(env_df)
                )

            with col2:

                st.metric(
                    "Start Year",
                    start_year
                )

            with col3:

                st.metric(
                    "End Year",
                    end_year
                )

            # ------------------------------------------------
            # CO2
            # ------------------------------------------------

            co2_column = get_first_existing_column(
                env_df,
                [
                    "co2",
                    "CO2",
                    "co2_emissions",
                ]
            )

            if co2_column is not None:

                st.subheader(
                    "🏭 India CO₂ Emission Trend"
                )

                chart = (
                    env_df
                    .set_index("year")
                    [[co2_column]]
                    .rename(
                        columns={
                            co2_column:
                            "CO₂ Emissions"
                        }
                    )
                )

                st.line_chart(
                    chart
                )

            # ------------------------------------------------
            # GHG
            # ------------------------------------------------

            ghg_column = get_first_existing_column(
                env_df,
                [
                    "total_ghg",
                    "ghg",
                    "Total_GHG",
                ]
            )

            if ghg_column is not None:

                st.subheader(
                    "🌱 Total Greenhouse Gas Emissions"
                )

                chart = (
                    env_df
                    .set_index("year")
                    [[ghg_column]]
                    .rename(
                        columns={
                            ghg_column:
                            "Total GHG"
                        }
                    )
                )

                st.line_chart(
                    chart
                )

            # ------------------------------------------------
            # PRIMARY ENERGY
            # ------------------------------------------------

            energy_column = get_first_existing_column(
                env_df,
                [
                    "primary_energy_consumption",
                    "Primary_Energy_Consumption",
                    "energy_consumption",
                ]
            )

            if energy_column is not None:

                st.subheader(
                    "⚡ Primary Energy Consumption"
                )

                chart = (
                    env_df
                    .set_index("year")
                    [[energy_column]]
                    .rename(
                        columns={
                            energy_column:
                            "Primary Energy Consumption"
                        }
                    )
                )

                st.line_chart(
                    chart
                )

            # ------------------------------------------------
            # RENEWABLE ENERGY
            # ------------------------------------------------

            renewable_column = get_first_existing_column(
                env_df,
                [
                    "renewables_consumption",
                    "renewable_energy",
                    "Renewable_Energy_Consumption",
                ]
            )

            if renewable_column is not None:

                st.subheader(
                    "🌱 Renewable Energy Consumption"
                )

                chart = (
                    env_df
                    .set_index("year")
                    [[renewable_column]]
                    .rename(
                        columns={
                            renewable_column:
                            "Renewable Energy"
                        }
                    )
                )

                st.line_chart(
                    chart
                )

            # ------------------------------------------------
            # FOSSIL FUEL
            # ------------------------------------------------

            fossil_column = get_first_existing_column(
                env_df,
                [
                    "fossil_fuel_consumption",
                    "Fossil_Fuel_Consumption",
                    "fossil_fuels",
                ]
            )

            if fossil_column is not None:

                st.subheader(
                    "⛽ Fossil Fuel Consumption"
                )

                chart = (
                    env_df
                    .set_index("year")
                    [[fossil_column]]
                    .rename(
                        columns={
                            fossil_column:
                            "Fossil Fuel Consumption"
                        }
                    )
                )

                st.line_chart(
                    chart
                )

            # ------------------------------------------------
            # TEMPERATURE + CO2 COMPARISON
            # ------------------------------------------------

            temperature_column = get_first_existing_column(
                env_df,
                [
                    "Mean_Temperature",
                    "mean_temperature",
                    "Temperature",
                ]
            )

            co2_per_capita_column = get_first_existing_column(
                env_df,
                [
                    "co2_per_capita",
                    "CO2_per_capita",
                ]
            )

            if (
                temperature_column is not None
                and co2_per_capita_column is not None
            ):

                st.subheader(
                    "📈 Temperature and CO₂ Comparison"
                )

                comparison_df = env_df[
                    [
                        "year",
                        temperature_column,
                        co2_per_capita_column,
                    ]
                ].copy()

                comparison_df = (
                    comparison_df
                    .set_index("year")
                    .rename(
                        columns={
                            temperature_column:
                            "Mean Temperature",
                            co2_per_capita_column:
                            "CO₂ per Capita",
                        }
                    )
                )

                min_values = (
                    comparison_df.min()
                )

                max_values = (
                    comparison_df.max()
                )

                denominator = (
                    max_values
                    - min_values
                )

                denominator = denominator.replace(
                    0,
                    1
                )

                normalized = (
                    comparison_df
                    - min_values
                ) / denominator

                st.line_chart(
                    normalized
                )

                st.caption(
                    "The comparison is normalized for visualization "
                    "because temperature and CO₂-per-capita values "
                    "have different units. This comparison does not "
                    "establish causation."
                )

            # ------------------------------------------------
            # FULL INTEGRATED DATA
            # ------------------------------------------------

            st.subheader(
                "📋 Integrated Environmental Dataset"
            )

            st.dataframe(
                env_df,
                use_container_width=True
            )

    except Exception as error:

        st.error(
            f"Environmental analysis failed: {error}"
        )


# ============================================================
# ESG ANALYSIS
# ============================================================

elif page == "ESG Analysis":

    st.header(
        "🌱 ESG & Sustainability Report Analysis"
    )

    st.write(
        "Upload a text-based sustainability report PDF "
        "for project-defined ESG keyword analysis."
    )

    uploaded_report = st.file_uploader(
        "Upload Sustainability Report",
        type=["pdf"],
        key="esg_pdf_uploader",
    )

    if uploaded_report is None:

        st.info(
            "No sustainability report uploaded yet."
        )

    else:

        st.success(
            "Sustainability report uploaded successfully."
        )

        try:

            report_text = extract_text_from_pdf(
                uploaded_report
            )

            if report_text.strip():

                current_esg_result = analyze_esg_text(
                    report_text
                )

                st.session_state.esg_result = (
                    current_esg_result
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "ESG Keyword Score",
                        (
                            f"{current_esg_result['esg_score']:.2f}"
                            " / 100"
                        )
                    )

                with col2:

                    st.metric(
                        "Keywords Found",
                        current_esg_result[
                            "total_keywords_found"
                        ]
                    )

                st.subheader(
                    "📌 ESG Assessment"
                )

                st.info(
                    current_esg_result[
                        "assessment"
                    ]
                )

                st.subheader(
                    "🔎 Detected ESG Topics"
                )

                matched_keywords = (
                    current_esg_result[
                        "matched_keywords"
                    ]
                )

                if matched_keywords:

                    for keyword in matched_keywords:

                        st.write(
                            f"✅ {keyword.title()}"
                        )

                else:

                    st.warning(
                        "No defined ESG keywords were detected."
                    )

                st.subheader(
                    "📄 Extracted Report Text"
                )

                st.text_area(
                    "Report Content",
                    report_text,
                    height=350,
                )

                st.caption(
                    "The ESG score is a project-defined keyword "
                    "coverage score. It is not an official ESG "
                    "rating, certification, or external ESG score."
                )

            else:

                st.warning(
                    "No readable text was found in this PDF."
                )

        except Exception as error:

            st.error(
                f"Unable to analyze the PDF: {error}"
            )


# ============================================================
# REPORT
# ============================================================

elif page == "Report":

    if df is None:

        st.warning(
            "Please provide a valid climate dataset."
        )

    else:

        st.header(
            "📄 Generate Final Analysis Report"
        )

        st.write(
            "Generate a complete report containing "
            "machine-learning, anomaly, risk, environmental, "
            "and ESG results."
        )

        try:

            # ------------------------------------------------
            # TRAIN MODELS
            # ------------------------------------------------

            (
                linear_model,
                linear_mae,
                linear_r2,
                y_test,
                linear_predictions,
            ) = train_temperature_model(
                df
            )

            (
                rf_model,
                rf_mae,
                rf_r2,
                _,
                rf_predictions,
            ) = train_random_forest_model(
                df
            )

            # ------------------------------------------------
            # ANOMALIES
            # ------------------------------------------------

            anomaly_data = detect_anomalies(
                df
            )

            anomaly_count = int(
                (
                    anomaly_data["Anomaly_Status"]
                    == "Anomaly"
                ).sum()
            )

            # ------------------------------------------------
            # RISK
            # ------------------------------------------------

            risk_data = calculate_risk_score(
                df
            )

            average_risk = (
                risk_data["Climate_Risk_Score"]
                .mean()
            )

            highest_risk = risk_data.loc[
                risk_data["Climate_Risk_Score"].idxmax(),
                "Risk_Category"
            ]

            # ------------------------------------------------
            # ENVIRONMENTAL DATA
            # ------------------------------------------------

            environmental_start_year = None
            environmental_end_year = None

            latest_co2 = None
            latest_ghg = None
            latest_energy = None
            latest_renewable = None
            latest_fossil = None

            env_df = load_environmental_data()

            if (
                env_df is not None
                and not env_df.empty
            ):

                environmental_start_year = int(
                    env_df["year"].min()
                )

                environmental_end_year = int(
                    env_df["year"].max()
                )

                env_df = env_df.sort_values(
                    "year"
                )

                # CO2
                co2_column = get_first_existing_column(
                    env_df,
                    [
                        "co2",
                        "CO2",
                        "co2_emissions",
                    ]
                )

                if co2_column is not None:

                    latest_co2 = safe_numeric(
                        env_df[co2_column]
                    ).iloc[-1]

                # GHG
                ghg_column = get_first_existing_column(
                    env_df,
                    [
                        "total_ghg",
                        "ghg",
                        "Total_GHG",
                    ]
                )

                if ghg_column is not None:

                    latest_ghg = safe_numeric(
                        env_df[ghg_column]
                    ).iloc[-1]

                # ENERGY
                energy_column = get_first_existing_column(
                    env_df,
                    [
                        "primary_energy_consumption",
                        "Primary_Energy_Consumption",
                        "energy_consumption",
                    ]
                )

                if energy_column is not None:

                    latest_energy = safe_numeric(
                        env_df[energy_column]
                    ).iloc[-1]

                # RENEWABLE
                renewable_column = get_first_existing_column(
                    env_df,
                    [
                        "renewables_consumption",
                        "renewable_energy",
                        "Renewable_Energy_Consumption",
                    ]
                )

                if renewable_column is not None:

                    latest_renewable = safe_numeric(
                        env_df[renewable_column]
                    ).iloc[-1]

                # FOSSIL
                fossil_column = get_first_existing_column(
                    env_df,
                    [
                        "fossil_fuel_consumption",
                        "Fossil_Fuel_Consumption",
                        "fossil_fuels",
                    ]
                )

                if fossil_column is not None:

                    latest_fossil = safe_numeric(
                        env_df[fossil_column]
                    ).iloc[-1]

            # ------------------------------------------------
            # ESG
            # ------------------------------------------------

            saved_esg_result = (
                st.session_state.get(
                    "esg_result"
                )
            )

            esg_score = None
            esg_assessment = None

            if saved_esg_result is not None:

                esg_score = saved_esg_result.get(
                    "esg_score"
                )

                esg_assessment = saved_esg_result.get(
                    "assessment"
                )

            # ------------------------------------------------
            # GENERATE REPORT
            # ------------------------------------------------

            report_html = generate_analysis_report(

                total_records=df.shape[0],

                total_features=df.shape[1],

                average_risk=average_risk,

                highest_risk=highest_risk,

                anomaly_count=anomaly_count,

                linear_mae=linear_mae,

                linear_r2=linear_r2,

                rf_mae=rf_mae,

                rf_r2=rf_r2,

                environmental_start_year=(
                    environmental_start_year
                ),

                environmental_end_year=(
                    environmental_end_year
                ),

                latest_co2=latest_co2,

                latest_ghg=latest_ghg,

                latest_energy=latest_energy,

                latest_renewable=latest_renewable,

                latest_fossil=latest_fossil,

                esg_score=esg_score,

                esg_assessment=esg_assessment,
            )

            # ------------------------------------------------
            # SAVE REPORT LOCALLY
            # ------------------------------------------------

            FINAL_REPORT_FILE.write_text(
                report_html,
                encoding="utf-8"
            )

            st.success(
                "Final report generated successfully."
            )

            st.info(
                "Report saved to:\n"
                f"{FINAL_REPORT_FILE}"
            )

            # ------------------------------------------------
            # DOWNLOAD REPORT
            # ------------------------------------------------

            st.download_button(
                label="📥 Download Final Report",
                data=report_html,
                file_name=(
                    "AI_Climate_Analysis_Final_Report.html"
                ),
                mime="text/html",
            )

            st.caption(
                "The final report combines the currently "
                "available climate, machine-learning, anomaly, "
                "risk, environmental, and ESG analysis results."
            )

        except Exception as error:

            st.error(
                f"Unable to generate final report: {error}"
            )