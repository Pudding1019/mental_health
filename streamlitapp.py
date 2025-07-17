import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载模型
model_data = joblib.load("suicide_risk_model.joblib")

# 页面配置
st.set_page_config(page_title="Suicide Risk Prediction", layout="centered")
st.title("🧠 Suicide Risk Prediction")
st.markdown("Enter original indicators below to predict suicide risk level.")

# ===== 用户输入原始变量 =====
with st.form("input_form"):
    st.subheader("🔢 Input Original Variables")

    alcohol = st.slider("Alcohol Use Disorders (%)", 0.0, 15.0, 5.0)
    bipolar = st.slider("Bipolar Disorders (%)", 0.0, 15.0, 5.0)
    anxiety = st.slider("Anxiety Disorders (%)", 0.0, 15.0, 5.0)
    eating = st.slider("Eating Disorders (%)", 0.0, 15.0, 5.0)
    unemployment = st.slider("Unemployment Rate (%)", 0.0, 25.0, 5.0)
    dropout = st.slider("Adolescent Dropout Rate (%)", 0.0, 30.0, 10.0)
    gdp = st.number_input("GDP per Worker", min_value=10000, max_value=100000, value=40000, step=1000)
    psychiatrists = st.slider("Psychiatrists (per 10,000 population)", 0.1, 10.0, 2.0)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    # Step 1: 构造包含原始输入的 DataFrame
    raw_data = pd.DataFrame({
        'AlcoholUseDisorders': [alcohol],
        'BipolarDisorders': [bipolar],
        'AnxietyDisorders': [anxiety],
        'EatingDisorders': [eating],
        'Unemployment': [unemployment],
        'Adolescent_Dropout': [dropout],
        'GDP_per_Worker': [gdp],
        'Psychiatrists(per 10 000 population)': [psychiatrists]
    })

    # Step 2: 使用 PCA 得出 MentalHealth_PC1
    mental_health_features = ['BipolarDisorders', 'AnxietyDisorders', 'EatingDisorders']
    scaler = StandardScaler()
    mental_health_scaled = scaler.fit_transform(raw_data[mental_health_features])
    pca = PCA(n_components=1)
    raw_data['MentalHealth_PC1'] = pca.fit_transform(mental_health_scaled)

    # Step 3: 构造交互变量
    raw_data['EcoMental_Interaction'] = raw_data['Unemployment'] * raw_data['MentalHealth_PC1']
    raw_data['Healthcare_Interaction'] = raw_data['Psychiatrists(per 10 000 population)'] * raw_data['AlcoholUseDisorders']

    # Step 4: 选取模型要求的特征并标准化
    required_features = model_data['final_features']
    model_input = raw_data[required_features]
    scaled_input = model_data['scaler'].transform(model_input)

    # Step 5: 模型预测
    pred_value = model_data['model'].predict(scaled_input)[0]
    pred_level = pd.cut(
        [pred_value],
        bins=model_data['risk_bins'],
        labels=model_data['risk_labels'],
        include_lowest=True
    )[0]

    # Step 6: 展示预测结果
    st.subheader("🧾 Prediction Result")
    st.write(f"**Predicted Risk Value:** `{pred_value:.2f}`")
    st.write(f"**Risk Level:** 🎯 `{pred_level}`")

    # Optional: 显示全部变量（含自动构造部分）
    if st.checkbox("Show full data with engineered features"):
        st.dataframe(raw_data)


