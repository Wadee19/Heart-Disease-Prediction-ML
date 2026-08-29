from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("artifacts/model.joblib")

st.set_page_config(page_title="Heart Disease ML Demo", page_icon="❤️", layout="centered")
st.title("Heart Disease Prediction — ML Demo")
st.caption("Educational portfolio demo only. Not a medical device or diagnostic tool.")

if not MODEL_PATH.exists():
    st.warning("No trained model found. Run `python src/train.py` first to create artifacts/model.joblib.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    sex = st.selectbox("Sex (dataset encoding)", options=[0, 1])
    chest_pain = st.selectbox("Chest pain type", options=[1, 2, 3, 4], index=3)
    bp = st.number_input("Resting blood pressure", min_value=70, max_value=250, value=130)
    cholesterol = st.number_input("Cholesterol", min_value=80, max_value=700, value=240)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", options=[0, 1])
    ekg = st.selectbox("Resting ECG result", options=[0, 1, 2])
    max_hr = st.number_input("Maximum heart rate", min_value=60, max_value=230, value=150)
    exercise_angina = st.selectbox("Exercise-induced angina", options=[0, 1])
    st_depression = st.number_input("ST depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST", options=[1, 2, 3])
    vessels = st.selectbox("Number of vessels (fluoroscopy)", options=[0, 1, 2, 3])
    thallium = st.selectbox("Thallium", options=[3, 6, 7])
    submitted = st.form_submit_button("Run prediction")

if submitted:
    sample = pd.DataFrame(
        [
            {
                "Age": age,
                "Sex": sex,
                "Chest pain type": chest_pain,
                "BP": bp,
                "Cholesterol": cholesterol,
                "FBS over 120": fbs,
                "EKG results": ekg,
                "Max HR": max_hr,
                "Exercise angina": exercise_angina,
                "ST depression": st_depression,
                "Slope of ST": slope,
                "Number of vessels fluro": vessels,
                "Thallium": thallium,
            }
        ]
    )
    probability = float(model.predict_proba(sample)[0, 1])
    prediction = int(probability >= 0.5)

    st.metric("Model probability of 'Presence'", f"{probability:.1%}")
    st.write("Predicted class:", "Presence" if prediction else "Absence")
    st.info("This output demonstrates the trained ML pipeline. It is not medical advice and must not be used for clinical decisions.")
