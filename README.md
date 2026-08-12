# Credit Card Fraud Prediction — Streamlit Deployment

## Required files
- app.py
- requirements.txt
- .python-version
- best_model.joblib
- scaler.joblib
- model_columns.joblib

## Streamlit Cloud
Set the Main file path to `app.py`.

The model was trained from the supplied `creditcard.csv` using the notebook workflow:
- duplicate removal
- log1p transformation of `Amount`
- 80/20 train-test split, random_state=42
- StandardScaler fitted on training data
- SMOTE on the scaled training data only
- Random Forest with class_weight='balanced'

For deployment practicality, the saved Random Forest uses 20 trees rather than the default 100; this keeps the artifact compact and substantially reduces training time while retaining the same preprocessing and model family.

The resulting held-out test performance for this saved artifact was:
- Accuracy: 0.9995
- Class-1 precision: 0.9444
- Class-1 recall: 0.7556
- Class-1 F1: 0.8395
- ROC-AUC from predicted labels: 0.8777
