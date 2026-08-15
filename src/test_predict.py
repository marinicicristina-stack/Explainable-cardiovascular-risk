# Import the prediction function
# Importer la fonction de prédiction

from predict import predict_heart_disease


# Example patient data
# Exemple de données patient

patient = {
    "age": 58,
    "sex": 1,
    "cp": 3,
    "trestbps": 132,
    "chol": 224,
    "fbs": 0,
    "restecg": 2,
    "thalach": 173,
    "exang": 0,
    "oldpeak": 3.2,
    "slope": 1,
    "ca": 2,
    "thal": 7
}


# Run the prediction
# Exécuter la prédiction

result = predict_heart_disease(patient)

print("Prediction result:")
print(result)
