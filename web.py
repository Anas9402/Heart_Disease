import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('heart.pkl')

# Title
st.title("💓 Heart Disease Prediction App")

# Sidebar inputs
st.sidebar.header("🧑‍⚕️ Patient Information")

def user_input_features():
    Age = st.sidebar.slider('Age', 20, 100, 50)
    RestingBP = st.sidebar.slider('RestingBP', 80, 200, 120)
    Cholesterol = st.sidebar.slider('Cholesterol', 100, 600, 200)
    FastingBS = st.sidebar.selectbox('FastingBS (Blood Sugar > 120 mg/dl)', [0, 1])
    MaxHR = st.sidebar.slider('MaxHR', 60, 220, 150)
    Oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.0, 1.0)

    Sex = st.sidebar.selectbox('Sex', ['M', 'F'])
    ChestPainType = st.sidebar.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
    RestingECG = st.sidebar.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
    ExerciseAngina = st.sidebar.selectbox('Exercise Angina', ['Y', 'N'])
    ST_Slope = st.sidebar.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

    data = {
        'Age': Age,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'MaxHR': MaxHR,
        'Oldpeak': Oldpeak,
        'Sex': Sex,
        'ChestPainType': ChestPainType,
        'RestingECG': RestingECG,
        'ExerciseAngina': ExerciseAngina,
        'ST_Slope': ST_Slope
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Prediction button
if st.button("🔍 Predict Heart Disease"):
    prediction = model.predict(input_df)
    st.subheader("🩺 Prediction Result")
    st.success(f"Heart Disease Risk: **{prediction[0]}**")

# Footer
st.markdown("---")
st.markdown("📌 Developed by **Anas Athar**")