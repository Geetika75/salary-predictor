import streamlit as st
import pandas as pd
import numpy as np
import joblib

from streamlit_option_menu import option_menu

# Load trained model
model = joblib.load("salary_predictor_model.pkl")
feature_columns = joblib.load("model_features.pkl")
job_titles = joblib.load("job_titles.pkl")
job_titles.append("Others")  # Add "Other" before sorting
job_titles = sorted(job_titles, key=lambda x: (x != "Others", x))  # Keep "Other" always on top
# selected_job = st.selectbox('Job Title', job_titles, max_options=5) # Add custom fallback option


# Use the same columns used during training

st.title("💼 Salary Predictor App")

# --- Input fields ---
age = st.number_input("Age", min_value=18, max_value=70, step=1)
experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.5)
gender = st.selectbox("Gender", ["Female", "Male"])
education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
selected_job = st.selectbox("Job Title", job_titles)

if selected_job == "Others":
     custom_title = st.text_input("Please enter your job title:")
     job_title = custom_title.strip().title()
else:
     job_title = selected_job

if st.button("Predict Salary"):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'Age': age,
        'Years of Experience': experience,
        'Gender': gender,
        'Education Level': education,
        'Job Title': job_title
    }])

    # Encode like in training
    encoded_data = pd.get_dummies(input_data, columns=['Gender', 'Education Level', 'Job Title'], drop_first=False)
    encoded_data = encoded_data.reindex(columns=feature_columns, fill_value=0)

    st.write("Encoded input features:", encoded_data)

    # Predict
    predicted_salary = model.predict(encoded_data)[0]

    st.success(f"💰 Estimated Salary: ₹{predicted_salary:,.2f}")
