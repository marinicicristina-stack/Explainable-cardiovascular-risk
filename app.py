import streamlit as st

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
    This application uses a machine learning model to estimate the
    probability of heart disease based on clinical patient data.

    The prediction will be accompanied by explainability information
    to help understand which features influenced the model output.
    """
)

st.info(
    "This application is a research prototype and does not provide "
    "a medical diagnosis."
)
