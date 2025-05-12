import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('oa_drug_response_model.pkl')

# Load LabelEncoders
le_gender = joblib.load('gender_label_encoder.pkl')
le_drug_type = joblib.load('drug_type_label_encoder.pkl')
le_dosage_level = joblib.load('dosage_level_label_encoder.pkl')
le_activity_level = joblib.load('activity_level_label_encoder.pkl')
le_smoking_status = joblib.load('smoking_status_label_encoder.pkl')
le_alcohol_consumption = joblib.load('alcohol_consumption_label_encoder.pkl')
le_response = joblib.load("response_label_encoder.pkl")

st.title("Osteoarthritis Risk Prediction")

# User Inputs
age = st.number_input("Age", 20, 100)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", 10.0, 50.0)
oa_severity = st.number_input("OA Severity (1–4 (Kellgren-Lawrence grade))", 0.0, 10.0)
duration_of_oa = st.number_input("Duration of OA (in months)", 0, 240)
crp = st.number_input("CRP Level (0–10 mg/L)", 0.0, 100.0)
esr = st.number_input("ESR Level (0–100 mm/hr)", 0.0, 100.0)
drug_type = st.selectbox("Drug Type", ["NSAID", "Corticosteroid", "Glucosamine", "Physiotherapy"])
dosage_level = st.selectbox("Dosage Level", ["Low", "Medium", "High"])
treatment_duration = st.number_input("Treatment Duration (in months)", 0, 365)
activity_level = st.selectbox("Activity Level", ["Low", "Moderate", "High"])
diet_score = st.number_input("Diet Score (0-10)", 0.0, 10.0)
smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
alcohol_consumption = st.selectbox("Alcohol Consumption", ["Yes", "No"])

# Encode categorical data
gender_enc = le_gender.transform([gender])[0]
drug_type_enc = le_drug_type.transform([drug_type])[0]
dosage_level_enc = le_dosage_level.transform([dosage_level])[0]
activity_level_enc = le_activity_level.transform([activity_level])[0]
smoking_status_enc = le_smoking_status.transform([smoking_status])[0]
alcohol_enc = le_alcohol_consumption.transform([alcohol_consumption])[0]

# Combine all inputs in correct order
input_data = np.array([[
    age,
    gender_enc,
    bmi,
    oa_severity,
    duration_of_oa,
    crp,
    esr,
    drug_type_enc,
    dosage_level_enc,
    treatment_duration,
    activity_level_enc,
    diet_score,
    smoking_status_enc,
    alcohol_enc
]])

# Load response label encoder


# Prediction
if st.button("Predict Drug Response"):
    prediction_encoded = model.predict(input_data)[0]
    prediction_label = le_response.inverse_transform([prediction_encoded])[0]
    st.success(f"The predicted drug response is: **{prediction_label}**")
