import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ========== 页面配置 ==========
st.set_page_config(page_title="Suicide Risk Prediction", layout="centered")
st.title("🧠 Suicide Risk Prediction")
st.markdown("Enter original indicators below to predict suicide risk level.")

# ========== 加载模型 ==========
model_data = joblib.load("suicide_risk_model.joblib")

# ========== 用户输入表单 ==========
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

# ========== 提交后处理 ==========
if submitted:
    # Step 1: 构造原始数据 DataFrame
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

    # Step 2: 手动构造 MentalHealth_PC1（标准化后加权平均作为近似替代）
    mental_health_features = ['BipolarDisorders', 'AnxietyDisorders', 'EatingDisorders']
    mental_scaled = StandardScaler().fit_transform(raw_data[mental_health_features])
    mental_pc1 = PCA(n_components=1).fit(mental_scaled).transform(mental_scaled)  # ⚠️ 注意：临时估算PCA值，调试可用
    raw_data['MentalHealth_PC1'] = mental_pc1

    # Step 3: 构造交互特征
    raw_data['EcoMental_Interaction'] = raw_data['Unemployment'] * raw_data['MentalHealth_PC1']
    raw_data['Healthcare_Interaction'] = raw_data['Psychiatrists(per 10 000 population)'] * raw_data['AlcoholUseDisorders']

    # Step 4: 提取模型要求特征
    final_features = model_data['final_features']
    model_input = raw_data[final_features]

    # Step 5: 标准化并预测
    model_scaled = model_data['scaler'].transform(model_input)
    pred_value = model_data['model'].predict(model_scaled)[0]
    pred_level = pd.cut(
        [pred_value],
        bins=model_data['risk_bins'],
        labels=model_data['risk_labels'],
        include_lowest=True
    )[0]

    # Step 6: 显示结果
    st.subheader("🧾 Prediction Result")
    st.write(f"**Predicted Risk Value:** {pred_value:.2f}")
    st.write(f"**Risk Level:** 🎯 {pred_level}")

    if st.checkbox("Predict new data"):
        st.dataframe(raw_data)
