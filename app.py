import streamlit as st
import joblib
import pandas as pd
import os

# Load model
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "models", "attrition_model.pkl")

model = joblib.load(model_path)

# Load dataset template (IMPORTANT)
data_path = os.path.join(base_path, "data", "employee.csv")
df_template = pd.read_csv(data_path).drop("Attrition", axis=1)

st.title("Smart Employee Prediction System")

# User Inputs
age = st.slider("Age", 18, 60, 30)
income = st.number_input("Monthly Income", 1000, 100000, 5000)
distance = st.slider("Distance From Home", 1, 30, 5)
years = st.slider("Years At Company", 0, 40, 5)

overtime = st.selectbox("OverTime", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
business = st.selectbox("BusinessTravel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

if st.button("Predict Attrition"):

    # Take one full row from dataset
    row = df_template.iloc[0:1].copy()

    # Update only selected inputs
    row['Age'] = age
    row['MonthlyIncome'] = income
    row['DistanceFromHome'] = distance
    row['YearsAtCompany'] = years
    row['OverTime'] = overtime
    row['Gender'] = gender
    row['BusinessTravel'] = business

    # Predict
    pred = model.predict(row)

    if pred[0] == "Yes":
        st.error("Employee likely to Leave")
    else:
        st.success("Employee likely to Stay")