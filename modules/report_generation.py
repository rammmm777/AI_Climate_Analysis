from datetime import datetime


def generate_analysis_report(
    total_records,
    total_features,
    average_risk,
    highest_risk,
    anomaly_count,
    linear_mae,
    linear_r2,
    rf_mae,
    rf_r2,
    esg_score=None,
    esg_assessment=None,
    environmental_start_year=None,
    environmental_end_year=None,
    latest_co2=None,
    latest_ghg=None,
    latest_energy=None,
    latest_renewable=None,
    latest_fossil=None
):
    """
    Generate an HTML summary report for the
    AI-Powered Climate Analysis System.
    """

    report_date = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    # ==================================================
    # ESG SECTION
    # ==================================================

    esg_section = ""

    if esg_score is not None:

        esg_section = f"""
        <h2>ESG Analysis</h2>

        <p>
            <strong>ESG Keyword Score:</strong>
            {esg_score:.2f} / 100
        </p>

        <p>
            <strong>Assessment:</strong>
            {esg_assessment}
        </p>
        """

    # ==================================================
    # ENVIRONMENTAL SECTION
    # ==================================================

    environmental_section = ""

    if (
        environmental_start_year is not None
        and environmental_end_year is not None
    ):

        environmental_section = f"""
        <h2>Environmental Analysis</h2>

        <p>
            <strong>Environmental Data Period:</strong>
            {environmental_start_year}–{environmental_end_year}
        </p>

        <table>

            <tr>
                <th>Environmental Indicator</th>
                <th>Latest Available Value</th>
            </tr>

            <tr>
                <td>CO₂ Emissions</td>
                <td>{_format_value(latest_co2)}</td>
            </tr>

            <tr>
                <td>Total GHG Emissions</td>
                <td>{_format_value(latest_ghg)}</td>
            </tr>

            <tr>
                <td>Primary Energy Consumption</td>
                <td>{_format_value(latest_energy)}</td>
            </tr>

            <tr>
                <td>Renewable Energy Consumption</td>
                <td>{_format_value(latest_renewable)}</td>
            </tr>

            <tr>
                <td>Fossil Fuel Consumption</td>
                <td>{_format_value(latest_fossil)}</td>
            </tr>

        </table>

        <p>
            The environmental analysis combines annual
            India-level CO₂/GHG and energy indicators with
            aggregated Hyderabad climate observations.
        </p>
        """


    # ==================================================
    # COMPLETE HTML REPORT
    # ==================================================

    report = f"""
    <!DOCTYPE html>

    <html>

    <head>

        <meta charset="UTF-8">

        <title>
            AI-Powered Climate Analysis Report
        </title>

        <style>

            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
                color: #222222;
            }}

            h1 {{
                color: #1f4e79;
            }}

            h2 {{
                color: #2f6f4e;
                margin-top: 30px;
            }}

            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 15px;
                margin-bottom: 20px;
            }}

            th,
            td {{
                border: 1px solid #cccccc;
                padding: 10px;
                text-align: left;
            }}

            th {{
                background-color: #eeeeee;
            }}

            .note {{
                background-color: #f4f4f4;
                padding: 12px;
                border-left: 4px solid #2f6f4e;
            }}

        </style>

    </head>


    <body>

        <h1>
            AI-Powered Climate Analysis System
        </h1>

        <p>
            <strong>Report Generated:</strong>
            {report_date}
        </p>


        <h2>
            Dataset Summary
        </h2>

        <table>

            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>

            <tr>
                <td>Total Climate Records</td>
                <td>{total_records}</td>
            </tr>

            <tr>
                <td>Total Climate Features</td>
                <td>{total_features}</td>
            </tr>

        </table>


        <h2>
            Machine Learning Results
        </h2>

        <table>

            <tr>
                <th>Model</th>
                <th>MAE (°C)</th>
                <th>R² Score</th>
            </tr>

            <tr>
                <td>Linear Regression</td>
                <td>{linear_mae:.4f}</td>
                <td>{linear_r2:.4f}</td>
            </tr>

            <tr>
                <td>Random Forest</td>
                <td>{rf_mae:.4f}</td>
                <td>{rf_r2:.4f}</td>
            </tr>

        </table>


        <h2>
            Anomaly Detection
        </h2>

        <p>
            <strong>Anomalies Detected:</strong>
            {anomaly_count}
        </p>


        <h2>
            Climate Risk Assessment
        </h2>

        <p>
            <strong>Average Climate Risk Score:</strong>
            {average_risk:.2f} / 100
        </p>

        <p>
            <strong>Highest Risk Category:</strong>
            {highest_risk}
        </p>


        {environmental_section}


        {esg_section}


        <h2>
            Methodology Note
        </h2>

        <div class="note">

            <p>
                The main climate analysis uses monthly
                Hyderabad climate observations. Environmental
                CO₂/GHG and energy indicators are annual
                India-level data. The datasets are aligned
                by year for integrated environmental analysis.
            </p>

            <p>
                Climate-risk scores and categories are
                based on the project-defined risk methodology.
            </p>

            <p>
                The ESG score is a project-defined
                keyword-coverage indicator and is not an
                official ESG rating.
            </p>

        </div>


        <h2>
            Conclusion
        </h2>

        <p>
            The AI-Powered Climate Analysis System combines
            climate data preprocessing, exploratory analysis,
            machine-learning prediction, anomaly detection,
            climate-risk assessment, environmental analysis,
            and ESG document analysis to support structured
            climate-related assessment.
        </p>

    </body>

    </html>
    """

    return report


# ==================================================
# HELPER FUNCTION
# ==================================================

def _format_value(value):

    if value is None:
        return "Not available"

    try:
        return f"{float(value):,.2f}"

    except (TypeError, ValueError):
        return str(value)