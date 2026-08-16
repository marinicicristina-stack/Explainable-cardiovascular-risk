# Explainable Cardiovascular Disease Risk Prediction

## Overview

This project develops an explainable clinical decision-support web application for cardiovascular disease risk prediction.

The application uses machine learning to estimate the probability of heart disease from patient clinical data. In addition to the prediction, SHAP (SHapley Additive exPlanations) is used to explain how individual clinical features influence the model output.

> **Disclaimer:** This application is a research prototype and does not provide a medical diagnosis.

---

## Project Context

This work is part of the project **Multimodal AI for Cardiovascular Risk Prediction**, developed at **VNU – Ho Chi Minh City International University (HCMIU)**.

This repository focuses on **Sub-project 4: Development of the Clinical Decision-Support Web Application**.

---

## Clinical Features

The model uses 13 clinical variables:

| Feature | Description |
|---|---|
| `age` | Age |
| `sex` | Sex |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting ECG result |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia-related feature |

---

## Machine Learning Pipeline

The development workflow includes:

1. Dataset loading and exploration
2. Data cleaning and preprocessing
3. Train/test split
4. Model training
5. Model evaluation and comparison
6. Model explainability with SHAP
7. Model serialization
8. Integration into the Streamlit web application

Three machine-learning models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

---

## Model Performance

The following results were obtained on the test set:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.833 | 0.846 | 0.786 | 0.815 |
| Random Forest | **0.850** | **0.880** | 0.786 | **0.830** |
| XGBoost | 0.817 | 0.840 | 0.750 | 0.792 |

Based on these results, **Random Forest** was selected for integration into the web application.

---

## Explainable AI with SHAP

SHAP is used to improve the interpretability of the Random Forest model.

The application provides patient-level explanations showing which clinical variables push the model output toward:

- **Heart Disease**
- **No Heart Disease**

This makes the prediction more transparent and allows users to understand the behavior of the machine-learning model.

SHAP values describe the behavior of the model and should not be interpreted as causal medical effects.

---

## Web Application

The application was developed with **Streamlit**.

Users can:

- Enter the 13 clinical variables
- Generate a cardiovascular disease prediction
- View the predicted probability
- Examine the main SHAP contributions for the individual prediction
- Visualize the most influential features

The web interface connects directly to the trained Random Forest model stored in the repository.

---

## Repository Structure

```text
Explainable-cardiovascular-risk/
│
├── app.py
├── data/
├── docs/
├── models/
│   └── random_forest_model.pkl
├── notebooks/
├── results/
├── src/
│   ├── predict.py
│   └── test_predict.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Run the Application Locally

Clone the repository:

```bash
git clone https://github.com/marinicicristina-stack/Explainable-cardiovascular-risk.git
cd Explainable-cardiovascular-risk
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Joblib
- Git / GitHub

---

## Current Status

- [x] Data preprocessing
- [x] Exploratory analysis
- [x] Model training
- [x] Model comparison
- [x] Random Forest model selection
- [x] Confusion matrix evaluation
- [x] SHAP global explainability
- [x] SHAP individual explainability
- [x] Model serialization
- [x] Prediction pipeline
- [x] Streamlit patient input interface
- [x] Online prediction
- [x] Patient-level SHAP explanation
- [x] SHAP contribution visualization
- [x] Streamlit Cloud deployment

---

## Limitations

This application was developed as a research prototype.

The model was evaluated on a limited dataset and has not been clinically validated. Its predictions must therefore not be used as a substitute for professional medical judgment or clinical diagnosis.
