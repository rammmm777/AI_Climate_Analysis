# 🌍 AI-Powered Climate Analysis System

## 1. Project Overview

The AI-Powered Climate Analysis System is a Python and Streamlit-based application designed to analyze climate data using data analytics and machine-learning techniques.

The system provides:

- Climate dataset preprocessing
- Exploratory Data Analysis (EDA)
- One-month-ahead temperature prediction
- Anomaly detection
- Climate-risk assessment
- Integrated environmental analysis
- ESG sustainability-report analysis
- SQLite database storage
- Automated HTML report generation
- Downloadable CSV analysis results

---

## 2. Main Technologies

- Python 3.12.1
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Plotly
- SQLite
- pypdf
- Jupyter Notebook
- Visual Studio Code
- Git/GitHub

---

## 3. Machine Learning Models

### Temperature Prediction

Two regression models are implemented:

1. Linear Regression
2. Random Forest Regressor

The system performs one-month-ahead temperature prediction using previous-month climate variables.

### Anomaly Detection

Isolation Forest is used to identify statistically unusual climate observations.

### Risk Assessment

A project-defined climate-risk scoring methodology is used to calculate a normalized risk score and corresponding category.

---

## 4. Dataset

### Main Climate Dataset

The main dataset contains monthly climate observations for Hyderabad from 2000 to 2025.

Variables include:

- Date
- Temperature
- Minimum Temperature
- Maximum Temperature
- Rainfall
- Pressure
- Sunshine

### Environmental Dataset

Annual environmental analysis integrates climate information with India-level:

- CO₂ emissions
- Greenhouse-gas emissions
- Energy consumption
- Renewable energy consumption
- Fossil-fuel consumption

The integrated environmental dataset covers 2000–2024.

---

## 5. Project Structure

```text
AI_Climate_Analysis/
│
├── data/
│
├── models/
│
├── modules/
│   ├── anomaly_detection.py
│   ├── data_validation.py
│   ├── esg_analysis.py
│   ├── pdf_analysis.py
│   ├── prediction.py
│   ├── preprocessing.py
│   ├── report_generation.py
│   └── risk_assessment.py
│
├── outputs/
├── reports/
├── assets/
│
├── app.py
├── config.py
├── database.py
├── requirements.txt
├── .gitignore
└── README.md