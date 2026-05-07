import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details to predict the risk of heart disease.")

@st.cache_resource
def train_model():
    df = pd.read_csv("heart.csv")
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    return LogisticRegression().fit(X_train, y_train), scaler

model, scaler = train_model()

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (0-3)", [0, 1, 2, 3])

sex_val = 1 if sex == "Male" else 0

if st.button("Predict ❤️"):
    input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ High risk of Heart Disease. Please consult a doctor!")
    else:
        st.success("✅ Low risk of Heart Disease. Keep maintaining a healthy lifestyle!")
