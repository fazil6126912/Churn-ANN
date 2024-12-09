import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

model = tf.keras.models.load_model("model-r.h5")

with open("le.pkl","rb") as f:
    le = pickle.load(f)

with open("ohe.pkl","rb") as f:
    ohe = pickle.load(f)

with open("scaler-r.pkl","rb") as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction")

# User input
geography = st. selectbox( 'Geography' , ohe.categories_[0])
gender = st.selectbox( 'Gender',le.classes_)
age = st.slider( 'Age',18, 92)
balance = st.number_input( 'Balance' )
credit_score = st.number_input( 'Credit Score' )
exited = st.selectbox( 'Has Exited' , [0,1])
tenure= st.slider( 'Tenure',0, 10)
num_of_products = st.slider( 'Number of Products' ,1,4)
has_cr_card = st.selectbox( 'Has Credit Card' , [0,1])
is_active_member = st.selectbox( 'Is Active Member' ,[0,1])

geography = ohe.transform([[geography]]).toarray()
input_data=pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': 2,
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

input_data = pd.concat([input_data,pd.DataFrame(geography,columns= ohe.get_feature_names_out(['Geography']))], axis = 1)
input_data = scaler.transform(input_data)

prediction = model.predict(input_data)[0][0]
st.write("Estimated Salary is ",prediction)
