import joblib
import pandas as pd

# Load the trained Random Forest model
# Charger le modèle Random Forest entraîné
model = joblib.load("models/random_forest_model.pkl")

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
