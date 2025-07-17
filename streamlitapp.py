
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 加载模型包
model_data = joblib.load("suicide_risk_model.joblib")
model = model_data["model"]
scaler = model_data["scaler"]
risk_bins = model_data["risk_bins"]
risk_labels = model_data["risk_labels"]
final_features = model_data["final_features"]

st.title("🧠 Suicide Risk Prediction App")
st.write("Provide the following information to estimate the suicide mortality rate and risk level.")

# -----------------------------
# 用户输入界面（8个原始变量）
# -----------------------------
alcohol = st.slider("Alcohol Use Disorders (%)", 0.0, 15.0, 5.0)
unemployment = st.slider("Unemployment (%)", 0.0, 25.0, 5.0)
dropout = st.slider("Adolescent Dropout (%)", 0.0, 30.0, 10.0)
bipolar = st.slider("Bipolar Disorder (%)", 0.0, 10.0, 1.0)
anxiety = st.slider("Anxiety Disorders (%)", 0.0, 20.0, 5.0)
eating = st.slider("Eating Disorders (%)", 0.0, 10.0, 1.0)
gdp = st.number_input("GDP per Worker", min_value=10000.0, max_value=150000.0, value=40000.0)
psy_beds = st.slider("Psychiatric hospital beds (per 100 000)", 0.0, 100.0, 10.0)

if st.button("🔍 Predict"):
    # Step 1: Mental health PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # 本地重新标准化并做 PCA（保持一致）
    mh_array = np.array([[bipolar, anxiety, eating]])
    local_scaler = StandardScaler()
    mh_scaled = local_scaler.fit_transform(mh_array)  # 模拟训练数据一致处理
    pca = PCA(n_components=1)
    mh_pc1 = pca.fit_transform(mh_scaled)[0, 0]  # 仅用当前输入生成PC1

    # Step 2: 构建所有特征
    data = {
        'AlcoholUseDisorders': alcohol,
        'Unemployment': unemployment,
        'Adolescent_Dropout': dropout,
        'MentalHealth_PC1': mh_pc1,
        'GDP_per_Worker': gdp,
        'EcoMental_Interaction': unemployment * mh_pc1,
        'Healthcare_Interaction': psy_beds * alcohol
    }

    input_df = pd.DataFrame([data])[final_features]
    scaled_input = scaler.transform(input_df)

    # Step 3: 预测 & 分类
    prediction = model.predict(scaled_input)[0]
    risk_level = pd.cut([prediction], bins=risk_bins, labels=risk_labels, include_lowest=True)[0]

    st.success(f"📈 Predicted Suicide Mortality Rate: **{prediction:.2f}**")
    st.info(f"🏷️ Risk Level: **{risk_level}**")
