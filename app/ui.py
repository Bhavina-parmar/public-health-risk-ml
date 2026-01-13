import streamlit as st
import requests

st.set_page_config(page_title="Health Risk Predictor", layout="wide")

st.title("🌍 Public Health Risk Prediction")
st.write("Enter values below and get real-time risk prediction!")

API_URL = "http://127.0.0.1:5000/predict"

# Feature Names
FEATURES = [
    "Year",
    "Prevalence Rate (%)",
    "Incidence Rate (%)",
    "Mortality Rate (%)",
    "Population Affected",
    "Healthcare Access (%)",
    "Doctors per 1000",
    "Hospital Beds per 1000",
    "Average Treatment Cost (USD)",
    "Recovery Rate (%)",
    "DALYs",
    "Improvement in 5 Years (%)",
    "Per Capita Income (USD)",
    "Education Index",
    "Urbanization Rate (%)"
]

# Input fields
user_input = {}
st.subheader("📊 Input Data")

for feature in FEATURES:
    user_input[feature] = st.number_input(feature, value=0.0)

# Predict button
if st.button("Predict Risk"):
    try:
        response = requests.post(API_URL, json=user_input)
        if response.status_code == 200:
            pred = response.json()["prediction"]
            label = "🔴 HIGH RISK" if pred == 1 else "🟢 LOW RISK"
            st.success(f"Prediction: {label}")
        else:
            st.error(f"Error: {response.json()}")
    except Exception as e:
        st.error(f"❌ API unreachable: {e}")
