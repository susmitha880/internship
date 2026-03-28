import streamlit as st
import pickle
import numpy as np

# Load saved model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# App title
st.set_page_config(page_title="AI Medical Diagnosis", layout="centered")
st.title("🩺 AI-Powered Medical Diagnosis System")
st.write("Predict disease based on patient health parameters")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120)
bp = st.number_input("Blood Pressure", min_value=50, max_value=200)
glucose = st.number_input("Glucose Level", min_value=50, max_value=300)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)

# Prediction button
if st.button("Predict Disease"):
    input_data = np.array([[age, bp, glucose, bmi]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error(f"⚠ Disease Detected (Confidence: {probability[0][1]*100:.2f}%)")
    else:
        st.success(f"✅ No Disease Detected (Confidence: {probability[0][0]*100:.2f}%)")

# Disclaimer
st.markdown("---")
st.warning("⚠ This system is for educational purposes only. Consult a doctor for medical advice.")
