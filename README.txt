# SecureCard: Credit Card Fraud Detection

This repository contains the code and models for my BSc Thesis: **Credit Card Fraud Detection on Imbalanced Data: A Machine Learning-Based Approach** at the University of Debrecen.

## Project Overview
The system uses Logistic Regression, Random Forest, and XGBoost to detect fraudulent transactions. It addresses the severe class imbalance using SMOTE. The final models are deployed in an interactive Streamlit web application called SecureCard.

## Files in this Repository
* `fraud_detection_model.ipynb`: Jupyter Notebook containing data preprocessing, model training, and evaluation.
* `securecard_app.py`: The Streamlit web application code.
* `securecard_fraud_model.pkl`: The trained Random Forest model.
* `securecard_xgb_model.pkl`: The trained XGBoost model.
* `test_with_fraud.csv`: A sample dataset for testing the web application.

## How to Run the App
1. Install the required packages: `pip install -r requirements.txt`
2. Run the Streamlit app: `streamlit run securecard_app.py`