# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Suicide Risk Predictor", layout="centered")

# Load model and components
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("suicide_risk_model.joblib")

model_data = load_model()

# Title
st.title("🧠 Suicide Risk Prediction App")

st.markdown("""
This app predicts **suicide risk level** based on socio-economic and mental health factors.
Fill in the fields below and click **Predict** to get risk level.
""")

# User Input
with st.form("input_form"):
    st.subheader("Input Factors")

    # Non-mental-health features
    AlcoholUseDisorders = st.slider("Alcohol Use Disorders (%)", 0.0, 15.0, 5.0)
    Unemployment = st.slider("Unemployment Rate (%)", 0.0, 30.0, 10.0)
    Adolescent_Dropout = st.slider("Adolescent Dropout Rate (%)", 0.0, 50.0, 15.0)
    GDP_per_Worker = st.number_input("GDP per Worker", min_value=5000.0, max_value=100000.0, value=40000.0)
    Psychiatrists = st.slider("Psychiatrists (per 10,000)", 0.0, 10.0, 2.0)

    # Mental-health raw inputs
    st.subheader("Mental Health Related Factors")
    BipolarDisorder = st.slider("Bipolar Disorder (%)", 0.0, 10.0, 2.0)
    AnxietyDisorders = st.slider("Anxiety Disorders (%)", 0.0, 15.0, 5.0)
    EatingDisorders = st.slider("Eating Disorders (%)", 0.0, 5.0, 1.0)
    TotalMental = st.slider("Total % Population Affected by Mental Illness", 0.0, 25.0, 8.0)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    # Step 1: compute PCA score
    mh_df = pd.DataFrame([{
        'BipolarDisorder': BipolarDisorder,
        'AnxietyDisorders': AnxietyDisorders,
        'EatingDisorders': EatingDisorders,
        'TotalPercentageOfPopulation': TotalMental
    }])

    mh_scaled = model_data['mh_scaler'].transform(mh_df)
    MentalHealth_PC1 = model_data['pca'].transform(mh_scaled)[0][0]

    # Step 2: Construct feature DataFrame
    input_df = pd.DataFrame({
        'AlcoholUseDisorders': [AlcoholUseDisorders],
        'Unemployment': [Unemployment],
        'Adolescent_Dropout': [Adolescent_Dropout],
        'MentalHealth_PC1': [MentalHealth_PC1],
        'GDP_per_Worker': [GDP_per_Worker],
        'Psychiatrists(per 10 000 population)': [Psychiatrists],
    })

    # Step 3: Feature engineering
    input_df['EcoMental_Interaction'] = input_df['Unemployment'] * input_df['MentalHealth_PC1']
    input_df['Healthcare_Interaction'] = input_df['Psychiatrists(per 10 000 population)'] * input_df['AlcoholUseDisorders']

    # Step 4: Align with training features
    X_input = input_df[model_data['final_features']]

    # Step 5: Scaling
    X_scaled = model_data['scaler'].transform(X_input)

    # Step 6: Prediction
    pred_value = model_data['model'].predict(X_scaled)[0]
    risk_level = pd.cut(
        [pred_value],
        bins=model_data['risk_bins'],
        labels=model_data['risk_labels'],
        include_lowest=True
    )[0]

    # Step 7: Display Results
    st.subheader("🔎 Prediction Result")
    st.markdown(f"**Predicted Risk Value:** {pred_value:.2f}")
    st.markdown(f"**Risk Level:** :red[{risk_level}]")

    fig, ax = plt.subplots()
    ax.bar(model_data['risk_labels'], [int(risk_level == label) for label in model_data['risk_labels']])
    ax.set_title("Predicted Risk Category")
    ax.set_ylabel("Confidence (1=Predicted Category)")
    st.pyplot(fig)
