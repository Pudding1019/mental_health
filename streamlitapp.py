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
st.markdown("Please enter the following indicators to predict the suicide risk level.")

# ===== 用户输入 =====
with st.form("input_form"):
    st.subheader("🔢 Input Variables")

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
    # 构造 DataFrame
    input_data = pd.DataFrame({
        'AlcoholUseDisorders': [alcohol],
        'BipolarDisorders': [bipolar],
        'AnxietyDisorders': [anxiety],
        'EatingDisorders': [eating],
        'Unemployment': [unemployment],
        'Adolescent_Dropout': [dropout],
        'GDP_per_Worker': [gdp],
        'Psychiatrists(per 10 000 population)': [psychiatrists]
    })

    # 心理健康 PCA 降维
    mental_features = ['BipolarDisorders', 'AnxietyDisorders', 'EatingDisorders']
    scaler = StandardScaler()
    mental_scaled = scaler.fit_transform(input_data[mental_features])
    pca = PCA(n_components=1)
    input_data['MentalHealth_PC1'] = pca.fit_transform(mental_scaled)

    # 构造交互特征
    input_data['EcoMental_Interaction'] = input_data['Unemployment'] * input_data['MentalHealth_PC1']
    input_data['Healthcare_Interaction'] = input_data['Psychiatrists(per 10 000 population)'] * input_data['AlcoholUseDisorders']

    # 特征提取与缩放
    selected_features = input_data[model_data['final_features']]
    scaled_input = model_data['scaler'].transform(selected_features)

    # 模型预测
    pred_value = model_data['model'].predict(scaled_input)[0]
    pred_level = pd.cut(
        [pred_value],
        bins=model_data['risk_bins'],
        labels=model_data['risk_labels'],
        include_lowest=True
    )[0]

    # 显示结果
    st.subheader("🧾 Prediction Result")
    st.write(f"**Predicted Risk Value:** `{pred_value:.2f}`")
    st.write(f"**Risk Level:** 🎯 `{pred_level}`")

    # 可选显示完整数据表
    if st.checkbox("Show input data with engineered features"):
        st.dataframe(input_data)


