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
            #pred = response.json()["prediction"]
            result = response.json()
            pred = result["prediction"]
            confidence = result["confidence"]
            drifted = result.get("drift_detected", [])
            if drifted:
                st.warning("⚠️ Data Drift Detected in:")
                for f in drifted:
                    st.write(f"• {f}")
            else:
                st.success("✅ No data drift detected")

            label = "🔴 HIGH RISK" if pred == 1 else "🟢 LOW RISK"
            #st.success(f"Prediction: {label}")

            st.success(f"Prediction:{label}")
            st.info(f"Confidence:{confidence * 100:.2f}%")
            
            st.session_state["last_input"]=user_input
            st.session_state["last_prediction"]=pred
            if "last_prediction" in st.session_state:
                st.subheader("Was this prediction correct?")
                col1,col2=st.columns(2)
                with col1:
                    if st.button("correct"):
                        feedback_payload={
                            "input":st.session_state["last_input"],
                            "prediction":st.session_state["last_prediction"],
                            "correct":True
                        }
                        requests.post("http://127.0.0.1:5000/feedback",json=feedback_payload)
                        st.success("Thanks! Feedback saved.")
                with col2:
                    if st.button("Wrong"):
                        feedback_payload={
                            "input":st.session_state["last_input"],
                            "prediction":st.session_state["last_prediction"],
                            "correct":False
                        }
                        requests.post("http://127.0.0.1:5000/feedback",json=feedback_payload)
                        st.success("Thanks! Feedback saved.")
        

        else:
            st.error(f"Error: {response.json()}")
    
    except Exception as e:
        st.error(f"❌ API unreachable: {e}")
    if st.button("Show Feature Importance"):
        try:
            resp = requests.get("http://127.0.0.1:5000/feature-importance")
            data = resp.json()["feature_importance"]

            features = [x[0] for x in data]
            importances = [x[1] for x in data]

            st.bar_chart(
                data=dict(zip(features, importances))
            )

        except Exception as e:
            st.error(f"Failed to load feature importance: {e}")
