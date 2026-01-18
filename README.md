# 📊 Customer Churn Prediction Web App (Machine Learning)

## 📌 Project Overview
This project predicts whether a customer is likely to **churn (leave the service)** using machine learning.  

It includes:

- A complete ML pipeline built in **Google Colab**
- A trained model saved using `pickle`
- A **Streamlit web application** for real-time predictions

Businesses can use this system to identify at-risk customers and take proactive retention actions.

---

## 🚀 Features
- Interactive web interface using Streamlit  
- Real-time churn prediction  
- Probability score for each prediction  
- Handles categorical features using saved encoders  
- Uses trained Random Forest model  

---

## 🧠 Machine Learning Workflow
- Data exploration and analysis  
- Data preprocessing & feature encoding  
- Handling class imbalance using **SMOTE**  
- Model training & comparison  
- Model serialization (`model.pkl`, `encoders.pkl`)  
- Web app integration  

---

## 🤖 Models Used
- Logistic Regression  
- Decision Tree  
- Random Forest ✅ (best performing)

---

## 📊 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- ROC-AUC  

---

## 🛠️ Technologies Used
- Python  
- Google Colab  
- Streamlit  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Pickle  

---

## 🖥️ App Interface

The app allows users to enter:

- Customer demographics  
- Account information  
- Services subscribed  
- Payment & billing details  

Then predicts:

- ✅ Whether the customer will churn  
- ✅ Probability of churn  

---

## 📌 Key Insights

- Dataset is highly imbalanced → **SMOTE** improves learning  
- Contract type, tenure, and monthly charges are strong churn indicators  
- Random Forest provides stable and accurate results  

---


