
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json

st.set_page_config(
    page_title="HR Attrition Risk Simulator",
    page_icon=":briefcase:",
    layout="wide"
)

st.title("HR Attrition Risk Simulator")
st.write("Enter employee details to predict attrition risk and get actionable HR recommendations")

# Currency conversion rate
USD_TO_INR = 83

@st.cache_resource
def load_model():
    model = joblib.load('attrition_model.joblib')
    scaler = joblib.load('scaler.joblib')
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Employee Details")
    age = st.slider("Age", 18, 60, 30)
    distance = st.slider("Distance From Home (miles)", 1, 50, 10)
    education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

with col2:
    st.subheader("Job Details")
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    job_role = st.selectbox("Job Role", [
        "Sales Executive", "Sales Representative", "Research Scientist",
        "Laboratory Technician", "Human Resources", "Manager",
        "Manufacturing Director", "Healthcare Representative"
    ])
    job_level = st.slider("Job Level", 1, 5, 2)
    
    # Income with USD and INR display
    monthly_income_usd = st.slider("Monthly Income (USD $)", 1000, 20000, 5000, step=500)
    monthly_income_inr = monthly_income_usd * USD_TO_INR
    st.caption(f"Monthly Income (INR): ₹{monthly_income_inr:,.2f}")
    
    overtime = st.selectbox("Overtime", ["No", "Yes"])

with col3:
    st.subheader("Work Environment")
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    years_in_role = st.slider("Years in Current Role", 0, 20, 3)
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
    environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    performance_rating = st.slider("Performance Rating (1-4)", 1, 4, 3)

if st.button("Predict Attrition Risk", type="primary"):
    input_data = {
        'Age': age,
        'DistanceFromHome': distance,
        'Education': education,
        'JobLevel': job_level,
        'MonthlyIncome': monthly_income_usd,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_role,
        'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'EnvironmentSatisfaction': environment_satisfaction,
        'PerformanceRating': performance_rating,
        'NumCompaniesWorked': 1,
        'TotalWorkingYears': years_at_company + 2,
        'TrainingTimesLastYear': 2,
        'PercentSalaryHike': 15,
        'StockOptionLevel': 0,
        'DailyRate': 800,
        'HourlyRate': 50,
        'MonthlyRate': 10000,
        'Department_Sales': 1 if department == "Sales" else 0,
        'Department_Research & Development': 1 if department == "Research & Development" else 0,
        'Department_Human Resources': 1 if department == "Human Resources" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Gender_Female': 1 if gender == "Female" else 0,
        'MaritalStatus_Married': 1 if marital == "Married" else 0,
        'MaritalStatus_Single': 1 if marital == "Single" else 0,
        'MaritalStatus_Divorced': 1 if marital == "Divorced" else 0,
        'OverTime_Yes': 1 if overtime == "Yes" else 0,
        'OverTime_No': 1 if overtime == "No" else 0,
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if prediction == 1:
            st.error("## High Risk of Attrition")
            st.warning(f"Probability: {probability*100:.1f}%")
            st.write("This employee shows signs of potential attrition.")
        else:
            st.success("## Low Risk of Attrition")
            st.info(f"Probability: {probability*100:.1f}%")
            st.write("This employee appears stable.")

    st.subheader("Key Risk Factors")
    st.write("These are the factors most influencing this prediction:")
    
    risk_factors = {
        "Overtime": "Employee works overtime regularly",
        "Job Satisfaction": "Low satisfaction increases risk",
        "Work Life Balance": "Poor balance increases risk",
        "Years at Company": "3-5 year mark shows higher risk",
        "Monthly Income": "Below market rate increases risk"
    }
    
    for factor, description in risk_factors.items():
        st.write(f"- **{factor}:** {description}")

    st.subheader("HR Recommendations")
    st.write("Based on this prediction, consider:")
    
    if prediction == 1:
        st.write("1. Schedule a stay interview to understand concerns")
        st.write("2. Review compensation and growth opportunities")
        st.write("3. Offer mentorship or career development programs")
        st.write("4. Consider flexible work arrangements")
    else:
        st.write("1. Continue regular check-ins and engagement")
        st.write("2. Recognize and reward contributions")
        st.write("3. Maintain positive work environment")

st.divider()
st.caption("Model: Logistic Regression | Accuracy: 74.8% | Recall: 61.7% | Currency: USD and INR (1 USD = ₹83)")

# Currency note in sidebar
with st.sidebar:
    st.subheader("Currency Information")
    st.write("This app displays amounts in both currencies:")
    st.write("- **USD ($)** - Original dataset")
    st.write("- **INR (₹)** - Conversion at ₹83 = $1")
    st.write("")
    st.write("**Exchange Rate:** $1 USD = ₹83 INR")
