import streamlit as st
import pandas as pd

from src.predict import predict_heart_disease


# Page configuration
# Configuration de la page
st.set_page_config(
    page_title="Cardiovascular Risk Prediction",
    page_icon="❤️",
    layout="wide"
)


# Application title
# Titre de l'application
st.title("Cardiovascular Disease Risk Prediction")

st.subheader("Explainable Clinical Decision-Support System")

st.write(
    """
    This application uses a machine learning model to estimate the probability
    of heart disease based on clinical patient data.

    The prediction will be accompanied by explainability information to help
    understand which features influenced the model output.
    """
)

st.info(
    "This application is a research prototype and does not provide "
    "a medical diagnosis."
)


st.divider()


# Patient information form
# Formulaire des informations patient
st.header("Patient Clinical Data")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=58
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type (cp)",
        options=[1, 2, 3, 4]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (mmHg)",
        min_value=70,
        max_value=250,
        value=132
    )

    chol = st.number_input(
        "Serum Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=224
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    restecg = st.selectbox(
        "Resting ECG Result",
        options=[0, 1, 2]
    )


with col2:
    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=60,
        max_value=220,
        value=173
    )

    exang = st.selectbox(
        "Exercise-Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    oldpeak = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=3.2,
        step=0.1
    )

    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[1, 2, 3]
    )

    ca = st.selectbox(
        "Number of Major Vessels (ca)",
        options=[0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thal",
        options=[3, 6, 7]
    )


st.divider()


# Run prediction
# Exécuter la prédiction
if st.button("Predict Heart Disease", type="primary"):

    patient_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    result = predict_heart_disease(patient_data)

    predicted_class = result["predicted_class"]
    probability = result["heart_disease_probability"]

    st.header("Prediction Result")

    if predicted_class == 1:
        st.error("Prediction: Heart Disease")
    else:
        st.success("Prediction: No Heart Disease")

    st.metric(
        "Model Probability of Heart Disease",
        f"{probability * 100:.1f}%"
    )

    st.caption(
        "The displayed probability is the output of the machine learning "
        "model and should not be interpreted as a clinical diagnosis."
    )
