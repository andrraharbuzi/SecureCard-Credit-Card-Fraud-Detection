# SecureCard: Credit Card Fraud Detection on Imbalanced Data

A machine learning-based system for detecting fraudulent credit card transactions using the Kaggle European Credit Card Fraud Dataset.
The project includes both a training notebook (comparing Logistic Regression, Random Forest, and XGBoost) and an interactive Streamlit application.

## Project Highlights
- Binary classification of highly imbalanced transaction data.
- Application of Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance.
- Comparison of Logistic Regression, Random Forest, and XGBoost models.
- Streamlit app for interactive dataset upload, prediction, and visualization.
- Reproducible dependency setup with a single `requirements.txt`.

## App Features
The Streamlit app (`securecard_app.py`) currently includes:
- `Dashboard` tab
  - Summary metrics (Total transactions, Fraud detected, Fraud rate).
  - Heuristic risk status based on predicted fraud rate.
- `Fraud Detection` tab
  - Upload custom CSV datasets (requires 30 specific feature columns).
  - Load sample data or balanced data for testing.
  - Run predictions and view suspicious transactions sorted by probability.
  - Download prediction results as a CSV file.
- `Model Performance` tab
  - Reference evaluation metrics (Precision, Recall, F1, ROC-AUC).
  - Feature importance chart and cross-validation visualizations.
- `System Information` tab
  - Application overview, supported models, and system workflow.

## Tech Stack
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- XGBoost
- imbalanced-learn
- Matplotlib, Seaborn
- Joblib

## Repository Structure
```text
.
├── fraud_detection_model.ipynb
├── securecard_app.py
├── securecard_fraud_model.pkl
├── securecard_xgb_model.pkl
├── test_with_fraud.csv
├── requirements.txt
└── README.md
(dataset files are downloaded locally; not tracked in Git)
```

## Dataset
This repository does not track the large raw CSV files.
Download the European Credit Card Fraud Dataset locally and place it in your workspace before running notebook training.

Suggested source:
- Kaggle Dataset page: [https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Setup
### 1. Create and activate a virtual environment

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Run the App
```bash
streamlit run securecard_app.py
```

## Training / Experiment Notebook
Use the notebook below for model development and experimentation:
- `fraud_detection_model.ipynb`

## Notes
- This project is a BSc Thesis implementation and performs offline batch classification; it is not a live banking integration.
- Model predictions are probability estimates, not absolute guarantees of fraud.
- Large datasets are intentionally excluded from version control for GitHub compatibility.

## Author
Andrra Harbuzi
