# 💰 Credit-card-Fraud-detection 🔍
## Overview 

This project is a credit card fraud detection system using Logistic Regression, built with Streamlit for easy interaction. The model is trained on an imbalanced dataset using SMOTE to balance classes and evaluated using classification metrics and ROC-AUC curves.

## 📂 Dataset

The dataset used is creditcard.csv, sourced from Kaggle.

✦ It contains transactions labeled as:

0 → Non-Fraudulent transaction

1 → Fraudulent transaction

## 📌Features

✦ SMOTE to handle class imbalance,LogisticRegression for training

✦ Feature scaling with StandardScaler.

✦ Interactive UI using Streamlit.

✦ Fraud Time Analysis Graph for insights.

✦ Model validation using Stratified K-Fold Cross-Validation.

✦ Performance evaluation using Accuracy, Classification Report, and ROC-AUC.
      
## 📈Performance Metrics

✦ See the classification report and ROC-AUC graph in [fraud detection(credit card).ipynb](fraud%20detection(credit%20card).ipynb).

| Metric           |   Score   |
|-------------------- |  ----------- |
| **Accuracy**     |   96.6 %   |
| **F1-Score**     |   95 %   |
| **ROC-AUC**      |    94 %   |

## 🔧 Installation & Setup

1. Clone the Repository

       git clone https://github.com/Anughna04/Credit-Card-Fraud-detection.git
       cd Credit-Card-Fraud-detection

2. Install Required Libraries

       pip install streamlit pandas numpy matplotlib scikit-learn imbalanced-learn joblib

3. Run the Streamlit App

       streamlit run app.py

This will launch the fraud detection web app.

### 📧 For any queries, contact me at [anughnakandimalla11@gmail.com](anughnakandimalla11@gmail.com).

## Tools & Technologies Used

✦ Programming: Python

 ✦ Data Processing📊: Pandas, NumPy

 ✦ Data Visualization📉: Matplotlib

 ✦ Machine Learning: Logistic Regression

 ✦ Class Imbalance Handling: SMOTE (imbalanced-learn)

 ✦ Model Persistence: Joblib

 Web App Framework: Streamlit🌐


## 👩‍💻Author

Anughna




