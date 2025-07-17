import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 加载模型、Scaler 和 PCA
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# 页面标题
st.title("💡 Suicide Mortality Rate Prediction")
st.write("Enter the following indicators to predict the suicide mortality rate and assess the risk level.")

# 用户输入
alcohol = st.slider("Alcohol Use Disorders (%)", 0.0, 15.0, 5.0)
unemployment = st.slider("Unemployment (%)", 0.0, 25.0, 5.0)
dropout = st.slider("Adolescent Dropout (%)", 0.0, 30.0, 10.0)

bipolar = st.slider("Bipolar Disorder (%)", 0.0, 10.0, 1.0)
anxiety = st.slider("Anxiety Disorders (%)", 0.0, 20.0, 5.0)
eating = st.slider("Eating Disorders (%)", 0.0, 10.0, 1.0)

gdp = st.number_input("GDP per Worker", min_value=10000.0, max_value=150000.0, value=40000.0)
psychiatrists = st.slider("Psychiatrists per 10,000 Population", 0.0, 5.0, 1.0)

# 预测按钮
if st.button("🔍 Predict"):
    # 1. 心理健康数据标准化 + PCA
    mh_array = np.array([[bipolar, anxiety, eating]])
    mh_scaled = scaler.transform(mh_array)
    mh_pc1 = pca.transform(mh_scaled)[0, 0]

    # 2. 构建最终输入特征（保持和模型训练时一致）
    input_data = pd.DataFrame([{
        "AlcoholUseDisorders": alcohol,
        "Unemployment": unemployment,
        "Adolescent_Dropout": dropout,
        "MentalHealth_PC1": mh_pc1,
        "GDP_per_Worker": gdp
    }])

    # 3. 模型预测
    prediction = model.predict(input_data)[0]

    # 4. 风险等级判定（可按你实际标准调整）
    if prediction < 5:
        risk = "🟢 Low"
    elif prediction < 15:
        risk = "🟡 Medium"
    else:
        risk = "🔴 High"

    # 5. 显示结果
    st.success(f"✅ Predicted Suicide Mortality Rate: **{prediction:.2f}** per 100,000")
    st.info(f"📊 Risk Level: **{risk}**")
