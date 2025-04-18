import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and scaler
try:
    model = joblib.load("depression_model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("Model file not found. Please train and save the model as 'depression_model.pkl'")
    st.stop()

st.set_page_config(page_title="Student Depression Predictor", layout="centered")
st.title("🎓 Student Depression Risk Predictor")
st.markdown("Fill in the details below to check for risk of depression based on lifestyle and academic habits.")

# UI inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 15, 50, 21)
academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
work_pressure = st.slider("Work Pressure (1-5)", 0, 5, 0)
cgpa = st.slider("CGPA (0-10)", 0.0, 10.0, 7.0)
study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)
job_satisfaction = st.slider("Job Satisfaction (1-5)", 0, 5, 0)
sleep_duration = st.selectbox("Sleep Duration", ["Less than 5", "5-6", "7-8", "9-10"])
diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
degree = st.selectbox("Degree", ["BCA", "BSc", "BA", "B.Pharm", "M.Tech", "Other"])
suicidal_thoughts = st.selectbox("Ever had suicidal thoughts?", ["Yes", "No"])
study_hours = st.slider("Work/Study Hours per day", 0.0, 16.0, 4.0)
financial_stress = st.selectbox("Financial Stress Level (1-5)", [1, 2, 3, 4, 5])
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

# Encode inputs like training
input_dict = {
    "Gender": 1 if gender == "Male" else 0,
    "Age": age,
    "Academic Pressure": academic_pressure,
    "Work Pressure": work_pressure,
    "CGPA": cgpa,
    "Study Satisfaction": study_satisfaction,
    "Job Satisfaction": job_satisfaction,
    "Sleep Duration": {
        "Less than 5": 4,
        "5-6": 5.5,
        "7-8": 7.5,
        "9-10": 9.5
    }[sleep_duration],
    "Dietary Habits": {"Healthy": 0, "Moderate": 1, "Unhealthy": 2}[diet],
    "Degree": {"BCA": 0, "BSc": 1, "BA": 2, "B.Pharm": 3, "M.Tech": 4, "Other": 5}[degree],
    "Have you ever had suicidal thoughts ?": 1 if suicidal_thoughts == "Yes" else 0,
    "Work/Study Hours": study_hours,
    "Financial Stress": int(financial_stress),
    "Family History of Mental Illness": 1 if family_history == "Yes" else 0
}

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][prediction]
    
    if prediction == 1:
        st.error(f"⚠️ Likely at risk of depression. (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Not likely at risk of depression. (Confidence: {prob:.2%})")

    st.caption("🧠 This is a prediction tool. Please consult professionals for clinical advice.")
