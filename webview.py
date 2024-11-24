# import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Promotion Prediction App")

# read the dataset to fill list values
df = pd.read_csv('train_LZdllcl.csv')

# create input fields 
# categorical columns
department = st.selectbox("department", pd.unique(df['department']))
region = st.selectbox("region", pd.unique(df['region']))
education = st.selectbox("education", pd.unique(df['education']))
gender = st.selectbox("gender", pd.unique(df['gender']))
recruitment_channel = st.selectbox("recruitment_channel", pd.unique(df['recruitment_channel']))

# numerical columns
no_of_trainings = st.number_input("no_of_trainings")
age = st.number_input("age")
previous_year_rating = st.number_input("previous_year_rating")
length_of_service = st.number_input("length_of_service")
KPIs_met_80 = st.number_input("KPIs_met >80%")
awards_won = st.number_input("Enter awards_won?")
avg_training_score = st.number_input("avg_training_score")

# map the user inputs to respective columns
# convert the input values to dict
inputs = {
  "department": department,
  "region": region,
  "education": education,
  "gender": gender,
  "recruitment_channel": recruitment_channel,
  "no_of_trainings": no_of_trainings,
  "age": age,
  "previous_year_rating": previous_year_rating,
  "length_of_service": length_of_service,
  "KPIs_met >80%": KPIs_met_80,
  "awards_won?": awards_won,
  "avg_training_score": avg_training_score
}

# load the model from tje pickle file
model = joblib.load('wns_pipeline_model.pkl')

# on click
if st.button("Predict"):
    # load the pickle model
    X_input = pd.DataFrame(inputs,index=[0])
    # predict the target using the loaded model
    prediction = model.predict(X_input)
    # display the result
    st.write("The predicted value is:")
    st.write(prediction)
