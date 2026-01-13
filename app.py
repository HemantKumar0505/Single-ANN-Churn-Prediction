# import streamlit as st
# import numpy as np
# import pickle
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder
# import pandas as pd


# # Load saved artifacts
# model = tf.keras.models.load_model('model.h5')

# with open('scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)

# with open('label_encoder_gender.pkl', 'rb') as file:
#     label_encoder_gender = pickle.load(file)

# with open('onehot_encoder_geo.pkl', 'rb') as file:
#     onehot_encoder_geo = pickle.load(file)

# Geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
# Gender = st.selectbox("Gender",label_encoder_gender.classes_)
# Age = st.slider("Age", 18, 92, 40)
# Balance = st.number_input("Balance")
# CreditScore = st.number_input("Credit Score")
# EstimatedSalary = st.number_input("Estimated Salary")
# Tenure = st.slider("Tenure (Years)", 0, 10)
# NumOfProducts = st.slider("Number of Products",1,4)
# HasCrCard = st.selectbox("Has Credit Card?", [0, 1])
# IsActiveMember = st.selectbox("Is Active Member?", [0, 1])

# input_data = pd.DataFrame({
#     'CreditScore':[CreditScore],
#     'Gender':[label_encoder_gender.transform([Gender])[0]],
#     'Age':[Age],
#     'Tenure':[Tenure],
#     'Balance':[Balance],
#     'NumOfProducts':[NumOfProducts],
#     'HasCrCard':[HasCrCard],
#     'IsActiveMember':[IsActiveMember],
#     'EstimatedSalary':[EstimatedSalary]
# })

# geo_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()
# geo_encoded_df = pd.DataFrame(
#     geo_encoded,
#     columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
# )


# input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# input_data_scaled = scaler.transform(input_data)

# prediction = model.predict(input_data_scaled)
# prediction_proba = prediction[0][0]

# st.write(f'Churn probability: {prediction_proba:.2f}')

# if prediction_proba > 0.5:
#     st.write('The customer is likely to churn.')
# else:
#     st.write('The customer is not likely to churn.')
















import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd

# Load artifacts
model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

# UI
st.title("Customer Churn Prediction")
Geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
Gender = st.selectbox("Gender", label_encoder_gender.classes_)
Age = st.slider("Age", 18, 92, 40)
Balance = st.number_input("Balance", min_value=0.0)
CreditScore = st.number_input("Credit Score", min_value=0.0)
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0)
Tenure = st.slider("Tenure (Years)", 0, 10, 5)
NumOfProducts = st.slider("Number of Products", 1, 4, 1)
HasCrCard = st.selectbox("Has Credit Card?", [0, 1])
IsActiveMember = st.selectbox("Is Active Member?", [0, 1])

# Base input
input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Gender': [label_encoder_gender.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})

# Geography one-hot
geo_encoded = onehot_encoder_geo.transform(
    pd.DataFrame({'Geography': [Geography]})
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Merge
input_data = pd.concat([input_data.reset_index(drop=True),
                        geo_encoded_df], axis=1)

# Scale
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn probability: {prediction_proba:.3f}')

if prediction_proba > 0.5:
    st.error('🚨 The customer is likely to churn.')
else:
    st.success('✅ The customer is not likely to churn.')
