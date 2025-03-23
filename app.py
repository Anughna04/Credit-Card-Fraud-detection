import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Credit Card Fraud Detection")

#Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    #Ensure only trained feature columns are used (drop "Class" if present)
    df_features = df.drop(columns=["Class"], errors="ignore")
    df_features = df_features[feature_names]  
    df_scaled = scaler.transform(df_features)

    predictions = model.predict(df_scaled)
    df["Fraud Prediction"] = predictions

    #Count fraud transactions
    fraud_count = np.sum(predictions)
    st.write(f"**Total Fraudulent Transactions Detected:** {fraud_count}")

    #Filter fraudulent transactions
    fraud_transactions = df[df["Fraud Prediction"] == 1]

    if not fraud_transactions.empty:
        st.write("### 🔍 **Fraudulent Transactions Detected:**")
        st.dataframe(fraud_transactions)

        #Fraud Transactions by Time
        if "Time" in fraud_transactions.columns:
            st.write("### ⏳ **Fraud Transactions Over Time**")
            plt.figure(figsize=(10, 5))
            sns.histplot(fraud_transactions["Time"], bins=24, kde=True, color="red")
            plt.xlabel("Time (Seconds from First Transaction)")
            plt.ylabel("Number of Fraudulent Transactions")
            plt.title("Distribution of Fraudulent Transactions Over Time")
            st.pyplot(plt)
        
        #Fraud Transactions by Amount**
        if "Amount" in fraud_transactions.columns:
            st.write("### 💰 **Fraud Transactions by Amount**")
            plt.figure(figsize=(10, 5))
            sns.histplot(fraud_transactions["Amount"], bins=30, kde=True, color="blue")
            plt.xlabel("Transaction Amount ($)")
            plt.ylabel("Number of Fraudulent Transactions")
            plt.title("Distribution of Fraudulent Transactions by Amount")
            st.pyplot(plt)

        if "Amount" in fraud_transactions.columns:
            total_fraud_amount = fraud_transactions["Amount"].sum()
            st.write(f"**Total Fraud Amount:** ${total_fraud_amount:,.2f}")
    else:
        st.write("✅ No fraudulent transactions detected.")

