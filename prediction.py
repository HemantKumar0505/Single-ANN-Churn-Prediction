# import tensorflow as tf
# # from tensorflow.keras.models import load_model

# import pickle
# import pandas as pd
# import numpy as np
# model = tf.keras.models.load_model('model.h5')

# with open('label_encoder_gender.pkl','rb') as file:
#     label_encoder_gender = pickle.load(file)
# with open('onehot_encoder_geo.pkl','rb') as file:
#     label_encoder_geo = pickle.load(file)

# with open('scaler.pkl','rb') as file:
#     scaler = pickle.load(file)

# input_data = {
#     'CreditScore':500,
#     'Geography':'germany',
#     'Gender':'Male',
#     'Age' :55,
#     'Tenure':5,
#     'Balance':50000,
#     'NumOfproduct':4,
#     'HasCrCard':0,
#     'IsActiveMember':1,
#     'EstimateSalary':40000

# }

# geo_encoded = label_encoder_geo.transform(
#     [[input_data['Geography']]]
# ).toarray()

# geo_encoded_df = pd.DataFrame(
#     geo_encoded,
#     columns=label_encoder_geo.get_feature_names_out(['Geography'])
# )

# input_df = pd.DataFrame([input_data])

# input_df['Gender'] = label_encoder_gender.tranform(input_df['Gender'])

# input_df = pd.concat(
#     [input_df.drop('Geography',axis=1),geo_encoded_df],
#     axis=1
# )


# input_scaled = scaler.transform(input_df)

# prediction = model.predict(input_scaled)

# prediction_proba = prediction[0][0]

# if prediction_proba > 0.5:
#     print('The customer is likely to churn.')
# else:
#     print('The customer is not likely to churn.')




import tensorflow as tf
import pickle
import pandas as pd

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    label_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

input_data = {
    'CreditScore': 500,
    'Geography': 'Germany',
    'Gender': 'Male',
    'Age': 55,
    'Tenure': 5,
    'Balance': 50000,
    'NumOfProducts': 4,
    'HasCrCard': 0,
    'IsActiveMember': 1,
    'EstimatedSalary': 40000
}

geo_encoded = label_encoder_geo.transform([[input_data['Geography']]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=label_encoder_geo.get_feature_names_out(['Geography'])
)

input_df = pd.DataFrame([input_data])

input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

input_df = pd.concat(
    [input_df.drop('Geography', axis=1), geo_encoded_df],
    axis=1
)

input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    print('The customer is likely to churn.')
else:
    print('The customer is not likely to churn.')
