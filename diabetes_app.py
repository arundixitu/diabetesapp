import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('xgb_model.pkl')  # Ensure the .pkl file is in the same folder as this script

# Add logo or image at the top
st.image(
    'diabetes_logo.png',
    caption="Diabetes Prediction App",
    use_container_width=False
)


# Title with emoji
st.title("🩺 Diabetes Prediction App")

# Add a horizontal separator
st.markdown("---")

# Sidebar content
st.sidebar.title("Navigation")
st.sidebar.markdown("Use this sidebar to explore the app.")
st.sidebar.markdown("## About")
st.sidebar.info("""
This app predicts the likelihood of diabetes using a machine learning model trained on the Pima Indians Diabetes dataset.
""")

st.sidebar.markdown("## Settings")
theme_choice = st.sidebar.radio(
    "Choose App Theme",
    ('Light', 'Dark', 'Custom')
)

# Create columns for better layout
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=120)

with col2:
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=130, value=70)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)

with col3:
    insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=80)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)

# Add sliders for remaining features
diabetes_pedigree_function = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
age = st.slider('Age', 0, 120, 25)

# Allow threshold selection with a slider
threshold = st.slider('Set Prediction Threshold (default is 0.5)', 0.1, 0.9, 0.5)

# Combine inputs into a feature array
inputs = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])

# Prediction logic
if st.button('Predict'):
    probability = model.predict_proba([inputs])[0][1]  # Get the probability of diabetes
    st.markdown(f"### Prediction Probability: {probability:.2f}")

    if probability > threshold:
        st.success('The patient is likely to have diabetes.')
    else:
        st.success('The patient is not having diabetes (unlikely).')

# Footer section
st.markdown("---")
st.markdown("### Thank you for using the Diabetes Prediction App!")
st.markdown("Feel free to [contact us](https://your-contact-link.com) for feedback or issues.")
