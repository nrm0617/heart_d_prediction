import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, and feature names
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

st.title("Heart Disease Prediction App")

# Numeric input fields
age = st.slider("Age", 20, 80, 50)
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 180, 120)
chol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 200)
thalach = st.slider("Maximum Heart Rate Achieved", 70, 200, 150)
oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)

# Categorical input fields (as one-hot keys)
sex = st.selectbox("Sex", ['sex_0', 'sex_1'])
cp = st.selectbox("Chest Pain Type", ['cp_0', 'cp_1', 'cp_2', 'cp_3'])
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['fbs_0', 'fbs_1'])
restecg = st.selectbox("Resting ECG Results", ['restecg_0', 'restecg_1', 'restecg_2'])
exang = st.selectbox("Exercise Induced Angina", ['exang_0', 'exang_1'])
slope = st.selectbox("Slope of Peak Exercise ST Segment", ['slope_0', 'slope_1', 'slope_2'])
ca = st.selectbox("Number of Major Vessels (0-3) Colored by Fluoroscopy", ['ca_0.0', 'ca_1.0', 'ca_2.0', 'ca_3.0'])
thal = st.selectbox("Thalassemia", ['thal_0.0', 'thal_1.0', 'thal_2.0'])

# Prepare input dictionary
input_dict = {
    'age': age,
    'trestbps': trestbps,
    'chol': chol,
    'thalach': thalach,
    'oldpeak': oldpeak,
    sex: 1,
    cp: 1,
    fbs: 1,
    restecg: 1,
    exang: 1,
    slope: 1,
    ca: 1,
    thal: 1
}

# Fill missing features with 0
for col in features:
    if col not in input_dict:
        input_dict[col] = 0

# Create DataFrame
input_df = pd.DataFrame([input_dict])

# Scale continuous features
scaled_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
input_df[scaled_cols] = scaler.transform(input_df[scaled_cols])

# Predict and display result
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
    st.success(f"Prediction Result: {result}")

