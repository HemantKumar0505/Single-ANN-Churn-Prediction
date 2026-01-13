import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder
import pandas as pd


# Load saved artifacts
model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

st.title("Bank Customer Churn Prediction (ANN)")

st.write("Enter details to check if customer is likely to churn 🔍")

CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.slider("Age", 18, 92, 40)
Tenure = st.slider("Tenure (Years)", 0, 10, 3)
Balance = st.number_input("Balance", min_value=0.0, value=60000.0)
NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4])
HasCrCard = st.selectbox("Has Credit Card?", [0, 1])
IsActiveMember = st.selectbox("Is Active Member?", [0, 1])
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

if st.button("Predict Churn"):
    # Encode gender
    Gender_encoded = label_encoder_gender.transform([Gender])[0]

    # Encode geography (one hot)
    geo_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()

    # Create input vector
    user_data = np.array([[CreditScore, Gender_encoded, Age, Tenure, Balance,
                           NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])

    # Concatenate with one hot geography
    user_data = np.concatenate([user_data, geo_encoded], axis=1)

    # Scale features
    user_data_scaled = scaler.transform(user_data)

    # Make prediction
    prob = model.predict(user_data_scaled)[0][0]
    
    churn = (prob > 0.5)

    if churn:
        st.error(f"⚠ Customer likely to CHURN!")
        st.write(f"Probability: **{prob*100:.2f}%**")
    else:
        st.success(f"😊 Customer likely to STAY!")
        st.write(f"Probability: **{prob*100:.2f}%**")
