"""
Thyroid Cancer Distant Metastasis Prediction System
Based on Clinical+3D_ITHscore Model (19 features)
With SHAP Visualization
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from datetime import datetime
import json
import shap
import matplotlib.pyplot as plt
from streamlit.components.v1 import html

# Page configuration
st.set_page_config(
    page_title="Thyroid Cancer Distant Metastasis Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Style settings
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .stButton>button {
        background-color: #5e72e4;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #4c63d2;
    }
    .result-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    h4 {
        color: #5e72e4;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Define all 19 features in the correct order
ALL_FEATURES = [
    'Sex', 'Age', 'BMI', 'Multifocal', 'Tumor_size', 
    'Number_of_metastatic_lymph_nodes', 'T_stage', 'WBC', 
    'RBC', 'PLT', 'HGB', 'MONO', 'NE', 'EOS', 
    'NLR', 'LMR', 'TG', 'TGAb', '3D_ITHscore'
]

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_loaded = False
    st.session_state.explainer = None

@st.cache_resource
def load_model():
    """Load XGBoost model from embedded base64 string"""
    import base64
    import tempfile
    import os
    
    # Base64 编码的模型（请将您的模型base64字符串粘贴在这里）
    model_base64 = """在此处粘贴您的base64编码字符串"""
    
    try:
        # 解码 base64
        model_bytes = base64.b64decode(model_base64)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.json', delete=False) as f:
            f.write(model_bytes)
            temp_path = f.name
        
        # 加载模型
        model = xgb.XGBClassifier()
        model.load_model(temp_path)
        
        # 删除临时文件
        os.unlink(temp_path)
        
        return model, True
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None, False

@st.cache_resource
def initialize_shap_explainer(model):
    """初始化SHAP解释器"""
    if model is not None:
        # 使用TreeExplainer，它专门为树模型优化
        explainer = shap.TreeExplainer(model)
        return explainer
    return None

def encode_categorical_features(features_dict):
    """Encode categorical features"""
    encoded = features_dict.copy()
    
    # Sex encoding (Male=1, Female=0)
    if 'Sex' in encoded:
        encoded['Sex'] = 1 if encoded['Sex'] == 'Male' else 0
    
    # Multifocal encoding (Yes=1, No=0)
    if 'Multifocal' in encoded:
        encoded['Multifocal'] = 1 if encoded['Multifocal'] == 'Yes' else 0
    
    # Tumor size encoding
    if 'Tumor_size' in encoded:
        size_map = {'≤1cm': 0, '1-2cm': 1, '>2cm': 2}
        encoded['Tumor_size'] = size_map.get(encoded['Tumor_size'], 0)
    
    # T stage encoding (T3/4=1, T1/2=0)
    if 'T_stage' in encoded:
        encoded['T_stage'] = 1 if encoded['T_stage'] == 'T3/4' else 0
    
    # TGAb encoding (≥115 IU/mL=1, <115 IU/mL=0)
    if 'TGAb' in encoded:
        encoded['TGAb'] = 1 if encoded['TGAb'] == '≥115 IU/mL' else 0
    
    return encoded

def predict_risk(model, features_df):
    """Predict risk using the model"""
    if model is not None:
        # Use real model for prediction
        probability = model.predict_proba(features_df)[0][1]
        return probability
    else:
        # Demo mode - return a warning message
        st.error("""
        ⚠️ **Demo Mode Active**
        
        The actual XGBoost model is not loaded. To get accurate predictions:
        1. Place the 'best_model_Clinical+3D_ITHscore.json' file in the app directory
        2. Or embed the model as base64 string in the code
        
        The model uses complex decision trees and feature interactions that cannot be 
        accurately simulated with simple weights.
        """)
        
        # Return the baseline prevalence from training data (13.5%)
        return 0.135

def generate_shap_force_plot(explainer, features_df, feature_values):
    """生成SHAP力图"""
    if explainer is None:
        return None, None, None
    
    # 计算SHAP值
    shap_values = explainer.shap_values(features_df)
    
    # 获取基准值（期望值）
    expected_value = explainer.expected_value
    
    # 如果是二分类，选择正类的SHAP值
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        expected_value = expected_value[1]
    
    # 创建特征名称映射，使用原始值而不是编码值
    feature_names = []
    for feature in ALL_FEATURES:
        if feature in feature_values:
            value = feature_values[feature]
            feature_names.append(f"{feature}={value}")
        else:
            feature_names.append(feature)
    
    return shap_values, expected_value, feature_names

def display_shap_force_plot(explainer, features_df, feature_values, probability):
    """显示SHAP力图"""
    try:
        shap_values, expected_value, feature_names = generate_shap_force_plot(
            explainer, features_df, feature_values
        )
        
        if shap_values is None:
            return
        
        # 使用matplotlib后端
        st.markdown("### 🎯 SHAP力图分析")
        st.markdown("展示各特征对预测结果的贡献度（红色=增加风险，蓝色=降低风险）")
        
        # 创建力图
        fig, ax = plt.subplots(figsize=(20, 3))
        shap.force_plot(
            expected_value,
            shap_values[0],
            features_df.iloc[0],
            feature_names=feature_names,
            out_names="远处转移风险",
            matplotlib=True,
            show=False,
            figsize=(20, 3)
        )
        st.pyplot(fig)
        plt.close()
        
        # 显示SHAP值详情
        st.markdown("#### 📊 特征贡献度详情")
        
        # 创建贡献度数据框
        shap_df = pd.DataFrame({
            '特征': feature_names,
            'SHAP值': shap_values[0],
            '特征值': features_df.iloc[0].values
        })
        
        # 按SHAP值绝对值排序
        shap_df['绝对SHAP值'] = abs(shap_df['SHAP值'])
        shap_df = shap_df.sort_values('绝对SHAP值', ascending=False)
        
        # 显示前10个最重要的特征
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 增加风险的特征（前5个）：**")
            risk_features = shap_df[shap_df['SHAP值'] > 0].head(5)
            for _, row in risk_features.iterrows():
                st.write(f"• {row['特征']}: +{row['SHAP值']:.3f}")
        
        with col2:
            st.markdown("**🔵 降低风险的特征（前5个）：**")
            protective_features = shap_df[shap_df['SHAP值'] < 0].head(5)
            for _, row in protective_features.iterrows():
                st.write(f"• {row['特征']}: {row['SHAP值']:.3f}")
        
        # 显示基准风险和实际预测
        st.info(f"""
        **模型预测解释：**
        - 基准风险（平均患者）: {expected_value:.1%}
        - SHAP值总和: {shap_values[0].sum():.3f}
        - 最终预测概率: {probability:.1%}
        
        *注：最终概率 = sigmoid(基准值 + SHAP值总和)*
        """)
        
        # 添加条形图
        display_shap_summary(explainer, features_df, feature_values)
        
    except Exception as e:
        st.error(f"SHAP可视化出错: {e}")

def display_shap_summary(explainer, features_df, feature_values):
    """显示SHAP摘要图（单个样本的条形图）"""
    if explainer is None:
        return
    
    try:
        st.markdown("### 📈 特征重要性条形图")
        
        # 计算SHAP值
        shap_values = explainer.shap_values(features_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # 创建条形图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 准备数据
        feature_names = []
        for feature in ALL_FEATURES:
            if feature in feature_values:
                value = feature_values[feature]
                feature_names.append(f"{feature}={value}")
            else:
                feature_names.append(feature)
        
        # 创建水平条形图
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP': shap_values[0]
        })
        shap_df = shap_df.sort_values('SHAP', key=abs)
        
        # 设置颜色
        colors = ['red' if x > 0 else 'blue' for x in shap_df['SHAP']]
        
        # 绘制条形图
        bars = ax.barh(shap_df['Feature'], shap_df['SHAP'], color=colors, alpha=0.7)
        
        # 添加数值标签
        for bar, value in zip(bars, shap_df['SHAP']):
            if value > 0:
                ax.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'+{value:.3f}', va='center', fontsize=8)
            else:
                ax.text(value - 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', va='center', ha='right', fontsize=8)
        
        ax.set_xlabel('SHAP值（对预测的贡献）')
        ax.set_title('特征对远处转移风险预测的贡献度')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close()
        
    except Exception as e:
        st.error(f"SHAP条形图生成出错: {e}")

# Main interface
st.title("🏥 Thyroid Cancer Distant Metastasis Prediction System")
st.markdown("### Clinical+3D_ITHscore Model (19 Features) with SHAP Analysis")

# Load model
model, loaded = load_model()
if loaded and model is not None:
    st.success("✅ Model loaded successfully")
    # 初始化SHAP解释器
    st.session_state.explainer = initialize_shap_explainer(model)
else:
    st.warning("⚠️ Using demo mode (model file not found)")
    st.session_state.explainer = None

# Instructions
with st.expander("📋 Instructions", expanded=False):
    st.info("""
    **Input Requirements:**
    - Clinical features (18): Patient demographics, tumor characteristics, blood markers
    - 3D_ITHscore (1): Quantitative imaging heterogeneity score (0-1)
    
    **Model Performance:**
    - Internal Test Set AUC-ROC: 0.746 (95% CI: 0.626-0.846)
    - External Validation AUC-ROC: 0.856 (95% CI: 0.739-0.943)
    - Optimal Threshold: 0.10
    - Specificity: 96.7% (internal), 94.6% (external)
    - Sensitivity: 25.0% (internal), 53.3% (external)
    
    **SHAP Analysis:**
    - 力图展示每个特征对预测的贡献
    - 红色条表示增加风险，蓝色条表示降低风险
    - 条的长度表示影响程度
    """)

# Create input form
st.markdown("---")
feature_values = {}

# Use columns for better layout
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 📝 Clinical Features")
    
    # Basic information
    st.markdown("#### Basic Information")
    bcol1, bcol2, bcol3 = st.columns(3)
    
    with bcol1:
        feature_values['Sex'] = st.selectbox("Sex", ["Female", "Male"])
    with bcol2:
        feature_values['Age'] = st.number_input("Age", min_value=0, max_value=120, value=45)
    with bcol3:
        feature_values['BMI'] = st.number_input("BMI", min_value=10.0, max_value=50.0, value=23.0, step=0.1)
    
    # Tumor characteristics
    st.markdown("#### Tumor Characteristics")
    tcol1, tcol2, tcol3 = st.columns(3)
    
    with tcol1:
        feature_values['Multifocal'] = st.selectbox("Multifocal", ["No", "Yes"])
    with tcol2:
        feature_values['Tumor_size'] = st.selectbox("Tumor size", ["≤1cm", "1-2cm", ">2cm"])
    with tcol3:
        feature_values['T_stage'] = st.selectbox("T stage", ["T1/2", "T3/4"])
    
    feature_values['Number_of_metastatic_lymph_nodes'] = st.number_input(
        "Number of metastatic lymph nodes", min_value=0, max_value=50, value=0
    )
    
    # Blood markers
    st.markdown("#### Blood Markers")
    
    # Row 1
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    with bcol1:
        feature_values['WBC'] = st.number_input("WBC (×10⁹/L)", min_value=2.0, max_value=20.0, value=6.0, step=0.1)
    with bcol2:
        feature_values['RBC'] = st.number_input("RBC (×10¹²/L)", min_value=3.0, max_value=6.0, value=4.5, step=0.1)
    with bcol3:
        feature_values['PLT'] = st.number_input("PLT (×10⁹/L)", min_value=100, max_value=500, value=250)
    with bcol4:
        feature_values['HGB'] = st.number_input("HGB (g/L)", min_value=80, max_value=200, value=130)
    
    # Row 2
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    with bcol1:
        feature_values['MONO'] = st.number_input("MONO", min_value=0.01, max_value=2.0, value=0.5, step=0.01)
    with bcol2:
        feature_values['NE'] = st.number_input("NE", min_value=1.0, max_value=15.0, value=4.0, step=0.1)
    with bcol3:
        feature_values['EOS'] = st.number_input("EOS", min_value=0.0, max_value=2.0, value=0.2, step=0.01)
    with bcol4:
        feature_values['NLR'] = st.number_input("NLR", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
    
    # Row 3
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        feature_values['LMR'] = st.number_input("LMR", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
    with bcol2:
        feature_values['TG'] = st.number_input("TG (ng/mL)", min_value=0.01, max_value=500.0, value=10.0, step=0.1)
    with bcol3:
        feature_values['TGAb'] = st.selectbox("TGAb", ["<115 IU/mL", "≥115 IU/mL"])

with col2:
    st.markdown("### 🔬 3D_ITHscore")
    st.info("3D_ITHscore is a quantitative measure of tumor heterogeneity derived from 3D imaging analysis. Range: 0-1, where higher values indicate greater heterogeneity.")
    
    feature_values['3D_ITHscore'] = st.slider(
        "3D_ITHscore",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Tumor heterogeneity score from 3D imaging analysis"
    )
    
    # Visual representation of ITHscore
    st.markdown("#### Heterogeneity Level")
    if feature_values['3D_ITHscore'] < 0.33:
        st.success("Low heterogeneity")
    elif feature_values['3D_ITHscore'] < 0.67:
        st.warning("Moderate heterogeneity")
    else:
        st.error("High heterogeneity")
    
    # Feature importance info
    st.markdown("#### Key Risk Factors")
    st.markdown("""
    **High-risk features in this model:**
    - T3/4 stage
    - Multiple lymph node metastases
    - Tumor size > 2cm
    - High 3D_ITHscore (>0.7)
    - Multifocal disease
    - Male sex
    - Age > 55 years
    """)

# Prediction button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Start Prediction", type="primary", use_container_width=True)

if predict_button:
    # Encode categorical features
    encoded_features = encode_categorical_features(feature_values)
    
    # Create feature DataFrame (maintain correct order)
    features_df = pd.DataFrame([encoded_features])[ALL_FEATURES]
    
    # Make prediction
    with st.spinner("Analyzing data..."):
        probability = predict_risk(model, features_df)
    
    # Display results
    st.markdown("---")
    st.markdown("### 🎯 Prediction Results")
    
    # Risk level determination (using optimal threshold of 0.10)
    optimal_threshold = 0.10
    
    if probability < optimal_threshold:
        risk_level = "Low Risk"
        risk_color = "#28a745"
        recommendation = """
        ✅ **Clinical Recommendations:**
        - Routine follow-up is sufficient
        - Annual neck ultrasound and thyroglobulin monitoring
        - Maintain healthy lifestyle
        - No immediate additional imaging required
        """
    elif probability < 0.3:
        risk_level = "Moderate Risk"
        risk_color = "#ffc107"
        recommendation = """
        ⚠️ **Clinical Recommendations:**
        - Enhanced surveillance recommended
        - Follow-up every 6 months
        - Consider chest CT for baseline assessment
        - Monitor thyroglobulin and anti-thyroglobulin antibodies closely
        - May benefit from additional risk stratification
        """
    else:
        risk_level = "High Risk"
        risk_color = "#dc3545"
        recommendation = """
        🚨 **Clinical Recommendations:**
        - Comprehensive evaluation urgently recommended
        - Consider PET-CT scan to detect distant metastasis
        - Chest CT and bone scan indicated
        - Multidisciplinary team consultation advised
        - Consider aggressive treatment strategies
        - Close monitoring with imaging every 3-4 months
        """
    
    # Results display
    st.markdown(f"""
    <div class="result-container">
        <h1 style="margin: 0; font-size: 3em;">{probability:.1%}</h1>
        <h2 style="color: white; margin: 10px 0;">Distant Metastasis Risk Probability</h2>
        <h2 style="color: {risk_color}; font-size: 2em;">{risk_level}</h2>
        <p style="color: white; margin-top: 20px;">
            Based on Clinical+3D_ITHscore Model<br>
            Optimal threshold: {optimal_threshold:.1%}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Risk Probability", f"{probability:.1%}")
    with col2:
        st.metric("Risk Level", risk_level)
    with col3:
        st.metric("Model Features", "19")
    
    # Clinical recommendations
    st.markdown("### 💡 Clinical Recommendations")
    st.markdown(recommendation)
    
    # SHAP可视化
    if st.session_state.get('explainer') is not None:
        st.markdown("---")
        display_shap_force_plot(
            st.session_state.explainer, 
            features_df, 
            feature_values, 
            probability
        )
    
    # Feature contribution analysis (原有的简单分析作为补充)
    st.markdown("---")
    st.markdown("### 🔍 Risk Factor Analysis (Rule-based)")
    
    # Create a simple risk factor summary
    risk_factors = []
    protective_factors = []
    
    if feature_values['T_stage'] == 'T3/4':
        risk_factors.append("Advanced T stage (T3/4)")
    if feature_values['Number_of_metastatic_lymph_nodes'] > 5:
        risk_factors.append(f"Multiple lymph node metastases ({feature_values['Number_of_metastatic_lymph_nodes']})")
    if feature_values['Tumor_size'] == '>2cm':
        risk_factors.append("Large tumor size (>2cm)")
    if feature_values['Multifocal'] == 'Yes':
        risk_factors.append("Multifocal disease")
    if feature_values['3D_ITHscore'] > 0.7:
        risk_factors.append(f"High tumor heterogeneity (ITHscore: {feature_values['3D_ITHscore']:.2f})")
    if feature_values['Sex'] == 'Male':
        risk_factors.append("Male sex")
    if feature_values['Age'] > 55:
        risk_factors.append(f"Age > 55 years ({feature_values['Age']})")
    
    if feature_values['T_stage'] == 'T1/2':
        protective_factors.append("Early T stage (T1/2)")
    if feature_values['Number_of_metastatic_lymph_nodes'] == 0:
        protective_factors.append("No lymph node metastases")
    if feature_values['3D_ITHscore'] < 0.3:
        protective_factors.append(f"Low tumor heterogeneity (ITHscore: {feature_values['3D_ITHscore']:.2f})")
    
    col1, col2 = st.columns(2)
    with col1:
        if risk_factors:
            st.error("**Risk Factors Present:**")
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.info("No major risk factors identified")
    
    with col2:
        if protective_factors:
            st.success("**Protective Factors:**")
            for factor in protective_factors:
                st.write(f"• {factor}")
        else:
            st.info("No protective factors identified")
    
    # Generate report
    st.markdown("### 📄 Assessment Report")
    
    report_content = f"""Thyroid Cancer Distant Metastasis Risk Assessment Report
=====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: Clinical+3D_ITHscore (19 features)

PATIENT INFORMATION
-----------------
Sex: {feature_values['Sex']}
Age: {feature_values['Age']} years
BMI: {feature_values['BMI']}

TUMOR CHARACTERISTICS
-------------------
Multifocal: {feature_values['Multifocal']}
Tumor size: {feature_values['Tumor_size']}
T stage: {feature_values['T_stage']}
Metastatic lymph nodes: {feature_values['Number_of_metastatic_lymph_nodes']}

LABORATORY VALUES
----------------
Blood counts:
- WBC: {feature_values['WBC']} ×10⁹/L
- RBC: {feature_values['RBC']} ×10¹²/L
- PLT: {feature_values['PLT']} ×10⁹/L
- HGB: {feature_values['HGB']} g/L

Cell differentials:
- MONO: {feature_values['MONO']}
- NE: {feature_values['NE']}
- EOS: {feature_values['EOS']}

Inflammatory markers:
- NLR: {feature_values['NLR']}
- LMR: {feature_values['LMR']}

Thyroid markers:
- TG: {feature_values['TG']} ng/mL
- TGAb: {feature_values['TGAb']}

IMAGING ANALYSIS
---------------
3D_ITHscore: {feature_values['3D_ITHscore']:.3f}
Heterogeneity level: {'Low' if feature_values['3D_ITHscore'] < 0.33 else 'Moderate' if feature_values['3D_ITHscore'] < 0.67 else 'High'}

RISK ASSESSMENT
--------------
Distant metastasis risk probability: {probability:.1%}
Risk level: {risk_level}
Model threshold used: {optimal_threshold:.1%}

Risk factors identified: {len(risk_factors)}
Protective factors identified: {len(protective_factors)}

CLINICAL RECOMMENDATIONS
-----------------------
{recommendation.replace('**', '').replace('✅', '').replace('⚠️', '').replace('🚨', '')}

DISCLAIMER
---------
This assessment is based on machine learning models and is for clinical reference only. 
Final diagnosis and treatment decisions should be made by professional physicians based 
on comprehensive patient evaluation.

Model Performance:
- Internal validation AUC-ROC: 0.746
- External validation AUC-ROC: 0.856
- Optimal threshold: 0.10

Physician signature: _____________
Date: _____________
"""
    
    # Download button
    st.download_button(
        label="📥 Download Report",
        data=report_content,
        file_name=f"thyca_risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

# Sidebar information
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.info(f"""
    **Model Status:** {'Loaded' if loaded else 'Demo Mode'}
    **SHAP Analysis:** {'Available' if st.session_state.explainer else 'Not Available'}
    **Model Type:** Clinical+3D_ITHscore
    **Total Features:** 19
    - Clinical: 18
    - 3D_ITHscore: 1
    
    **Performance (External):**
    - AUC-ROC: 0.856
    - Sensitivity: 53.3%
    - Specificity: 94.6%
    - F2-Score: 0.533
    - MCC: 0.480
    """)
    
    st.markdown("### 📝 Feature Categories")
    with st.expander("Clinical Features (18)"):
        st.markdown("""
        **Demographics (3):**
        - Sex, Age, BMI
        
        **Tumor (4):**
        - Multifocal, Tumor size
        - T stage, Lymph nodes
        
        **Blood counts (7):**
        - WBC, RBC, PLT, HGB
        - MONO, NE, EOS
        
        **Inflammatory (2):**
        - NLR, LMR
        
        **Thyroid markers (2):**
        - TG, TGAb
        """)
    
    with st.expander("3D_ITHscore"):
        st.markdown("""
        Quantitative measure of tumor 
        heterogeneity from 3D imaging:
        - Range: 0-1
        - Higher = more heterogeneous
        - Key predictor of metastasis
        """)
    
    st.markdown("### ⚠️ Usage Notice")
    st.warning("""
    1. For medical professionals only
    2. Not a substitute for clinical judgment
    3. Based on limited training data
    4. Regular model updates recommended
    """)
    
    st.markdown("### 🎯 SHAP Analysis")
    st.success("""
    SHAP (SHapley Additive exPlanations) 
    提供了模型可解释性：
    - 展示每个特征的贡献度
    - 红色增加风险，蓝色降低风险
    - 帮助理解模型决策过程
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Thyroid Cancer Distant Metastasis Prediction System v3.0</p>
    <p>Based on Clinical+3D_ITHscore XGBoost Model with SHAP Analysis</p>
    <p>© 2025 | For Medical Research and Clinical Reference Only</p>
</div>
""", unsafe_allow_html=True)








