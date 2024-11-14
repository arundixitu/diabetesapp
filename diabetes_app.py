import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('xgb_model.pkl')  # Ensure this file exists

# Title of the web app
st.title("Diabetes Prediction App")

# Input features
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=120)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=130, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=80)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input('Age', min_value=0, max_value=120, value=25)

# Threshold input
threshold = st.slider('Set Prediction Threshold (default is 0.5)', min_value=0.1, max_value=0.9, value=0.5)

# Prediction
inputs = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])
if st.button('Predict'):
    probability = model.predict_proba([inputs])[0][1]  # Get the probability of diabetes
    st.write(f"Prediction Probability: {probability:.2f}")
    if probability > threshold:
        st.success('The patient is likely to have diabetes.')
    else:
        st.success('The patient is unlikely to have diabetes.')
