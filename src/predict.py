import shap
import joblib
import pandas as pd

# Load the trained Random Forest model
# Charger le modèle Random Forest entraîné
model = joblib.load("models/random_forest_model.pkl")
# Create the SHAP explainer
# Créer l'explainer SHAP
explainer = shap.TreeExplainer(model)

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]
def predict_heart_disease(patient_data):
    """
    Predict heart disease for one patient.

    Parameters
    ----------
    patient_data : dict
        Dictionary containing the 13 clinical features.

    Returns
    -------
    dict
        Predicted class and model probability.
    """

    patient_df = pd.DataFrame([patient_data], columns=FEATURES)

    predicted_class = int(model.predict(patient_df)[0])
    probability = float(model.predict_proba(patient_df)[0, 1])

    return {
        "predicted_class": predicted_class,
        "heart_disease_probability": round(probability, 3)
    }


def explain_prediction(patient_data):
    """
    Explain the prediction for one patient using SHAP.

    Parameters
    ----------
    patient_data : dict
        Dictionary containing the 13 clinical features.

    Returns
    -------
    dict
        SHAP values for the Heart Disease class.
    """

    patient_df = pd.DataFrame([patient_data], columns=FEATURES)

    # Calculate SHAP values for the patient
    # Calculer les valeurs SHAP pour le patient
    shap_values = explainer(patient_df)

    # Extract SHAP values for class 1 (Heart Disease)
    # Extraire les valeurs SHAP pour la classe 1
    class_1_values = shap_values.values[0, :, 1]

    explanation = {
        feature: float(value)
        for feature, value in zip(FEATURES, class_1_values)
    }

    return explanation

