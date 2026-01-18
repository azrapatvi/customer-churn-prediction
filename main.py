import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.markdown(
    "Fill in the customer details below to predict whether the customer is likely to **churn or stay**."
)
st.divider()

st.subheader("👤 Customer Information")
gender = st.text_input("Enter Gender")
SeniorCitizen = st.selectbox("Senior Citizen (1 = Yes, 0 = No)", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, step=1)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)


predict = st.button("🔍 Predict Churn")

df=pd.DataFrame([{
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}])

if predict:

    with open('encoders.pkl','rb') as f:
        encoders=pickle.load(f)

    for col, encoder in encoders.items():
        df[col] = df[col].astype(str)
        df[col] = df[col].apply(
            lambda x: x if x in encoder.classes_ else encoder.classes_[0]
        )
        df[col] = encoder.transform(df[col])

    with open('model.pkl','rb') as f:
        model=pickle.load(f)

    prediction = model.predict(df)
    probability = model.predict_proba(df)

    # Output
    if prediction[0] == 1:
        st.error(f"Customer will CHURN (Probability: {probability[0][1]*100:.2f}%)")
    else:
        st.success(f"Customer will NOT churn (Probability: {probability[0][0]*100:.2f}%)")