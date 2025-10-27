import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load and preprocess data
df = pd.read_csv('heart.csv').drop_duplicates()
x = df.drop(['HeartDisease'], axis=1)
y = df['HeartDisease'].map({0: 'No', 1: 'Yes'})

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

num_attr = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
cat_attr = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False))
])

preprocessing = ColumnTransformer(transformers=[
    ('cat', cat_pipeline, cat_attr),
    ('num', num_pipeline, num_attr)
])

pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('model', KNeighborsClassifier(n_neighbors=5))
])

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

# Save model
joblib.dump(pipeline, 'heart.pkl')

# Streamlit App
st.title("💓 Heart Disease Prediction App")
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

if st.button("🔍 Predict Heart Disease"):
    model = joblib.load('heart.pkl')
    prediction = model.predict(input_df)
    st.subheader("🩺 Prediction Result")
    st.success(f"Heart Disease Risk: **{prediction[0]}**")

st.markdown("---")
st.markdown("📌 Developed by **Anas Athar**")