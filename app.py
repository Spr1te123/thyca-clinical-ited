"""
甲状腺癌远处转移预测系统 - 完整版
基于Clinical+iTED模型（40个特征）
支持手动输入所有特征值
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from datetime import datetime
import json

# 页面配置
st.set_page_config(
    page_title="甲状腺癌远处转移预测系统-完整版",
    page_icon="🏥",
    layout="wide"
)

# 样式设置
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

# 定义所有40个特征的配置（按照模型训练时的真实顺序）
ALL_FEATURES = [
    # 临床特征 (20个) - 注意：Sex在Age之前！
    'Sex', 'Age', 'BMI', 'Benign_thyroid_lesions', 'Multifocal', 
    'Tumor_size', 'Infiltrated_the_adjacent_tissue', 'Number_of_metastatic_lymph_nodes',
    'T_stage', 'WBC', 'HGB', 'MONO', 'NE', 'BASO', 'NLR', 'SII', 
    'TG', 'TGAb', 'TPOAb', 'GLU',
    
    # iTED特征 (20个)
    'iTED_firstorder_10Percentile', 'iTED_firstorder_Entropy', 'iTED_firstorder_Kurtosis',
    'iTED_firstorder_Median', 'iTED_firstorder_Range', 'iTED_firstorder_Uniformity',
    'iTED_firstorder_Variance', 'iTED_glcm_Correlation', 'iTED_glcm_Id',
    'iTED_glrlm_GrayLevelNonUniformityNormalized', 'iTED_glrlm_ShortRunHighGrayLevelEmphasis',
    'iTED_gldm_DependenceEntropy', 'iTED_gldm_DependenceVariance',
    'iTED_gldm_LargeDependenceEmphasis', 'iTED_gldm_LargeDependenceLowGrayLevelEmphasis',
    'iTED_gldm_SmallDependenceEmphasis', 'iTED_gldm_SmallDependenceLowGrayLevelEmphasis',
    'iTED_glszm_LargeAreaEmphasis', 'iTED_ngtdm_Coarseness', 'iTED_ngtdm_Contrast'
]

# 初始化session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_loaded = False

@st.cache_resource
def load_model():
    """加载XGBoost模型"""
    try:
        model = xgb.XGBClassifier()
        model.load_model('models/best_model_Clinical_iTED.json')
        return model, True
    except:
        # 如果加载失败，返回None（演示模式）
        return None, False

def encode_categorical_features(features_dict):
    """编码分类特征"""
    encoded = features_dict.copy()
    
    # 性别编码
    if 'Sex' in encoded:
        encoded['Sex'] = 1 if encoded['Sex'] == '男' else 0
    
    # 良性病变编码
    if 'Benign_thyroid_lesions' in encoded:
        encoded['Benign_thyroid_lesions'] = 1 if encoded['Benign_thyroid_lesions'] == '是' else 0
    
    # 多灶性编码
    if 'Multifocal' in encoded:
        encoded['Multifocal'] = 1 if encoded['Multifocal'] == '是' else 0
    
    # 肿瘤大小编码
    if 'Tumor_size' in encoded:
        size_map = {'≤1cm': 0, '1-2cm': 1, '>2cm': 2}
        encoded['Tumor_size'] = size_map.get(encoded['Tumor_size'], 0)
    
    # 邻近组织浸润编码
    if 'Infiltrated_the_adjacent_tissue' in encoded:
        encoded['Infiltrated_the_adjacent_tissue'] = 1 if encoded['Infiltrated_the_adjacent_tissue'] == '是' else 0
    
    # T分期编码
    if 'T_stage' in encoded:
        encoded['T_stage'] = 1 if encoded['T_stage'] == 'T3/4' else 0
    
    # 抗体编码
    if 'TGAb' in encoded:
        encoded['TGAb'] = 1 if encoded['TGAb'] == '≥115 IU/mL' else 0
    
    if 'TPOAb' in encoded:
        encoded['TPOAb'] = 1 if encoded['TPOAb'] == '≥40 IU/mL' else 0
    
    return encoded

def predict_risk(model, features_df):
    """使用模型预测风险"""
    if model is not None:
        # 使用真实模型预测
        probability = model.predict_proba(features_df)[0][1]
    else:
        # 演示模式 - 基于特征计算模拟概率
        features = features_df.iloc[0]
        risk_score = 0.1  # 基础风险
        
        # 临床特征权重
        if features.get('Sex', 0) == 1:  # 男性
            risk_score += 0.05
        if features.get('Age', 0) > 55:
            risk_score += 0.08
        if features.get('Multifocal', 0) == 1:
            risk_score += 0.10
        if features.get('Tumor_size', 0) == 2:
            risk_score += 0.15
        elif features.get('Tumor_size', 0) == 1:
            risk_score += 0.08
        if features.get('T_stage', 0) == 1:
            risk_score += 0.15
        if features.get('Number_of_metastatic_lymph_nodes', 0) > 5:
            risk_score += 0.15
        
        # iTED特征影响（模拟）
        if features.get('iTED_firstorder_Entropy', 0) > 0.8:
            risk_score += 0.10
        if features.get('iTED_glcm_Correlation', 0) < 0.1:
            risk_score += 0.08
        
        probability = min(risk_score, 0.95)
    
    return probability

# 主界面
st.title("🏥 甲状腺癌远处转移预测系统")
st.markdown("### 基于Clinical+iTED模型（40特征）的综合评估")

# 加载模型
model, loaded = load_model()
if loaded:
    st.success("✅ 模型加载成功")
else:
    st.warning("⚠️ 使用演示模式（模型文件未找到）")

# 说明
with st.expander("📋 使用说明", expanded=False):
    st.info("""
    **输入要求：**
    1. 临床特征（20个）：患者基本信息、肿瘤特征、血液学和生化指标
    2. iTED特征（20个）：从医学影像中提取的纹理特征
    
    **注意事项：**
    - 请确保所有数值准确
    - iTED特征通常由专业软件计算得出
    - 如不确定某项数值，请咨询相关科室
    
    **模型性能：**
    - AUC-ROC: 0.92
    - 敏感性: 85%
    - 特异性: 87%
    """)

# 创建输入表单
st.markdown("---")
feature_values = {}

# 使用标签页组织输入
tab1, tab2 = st.tabs(["📝 临床特征（1-20）", "🔬 iTED特征（21-40）"])

with tab1:
    st.markdown("### 临床特征输入")
    
    # 基本信息
    st.markdown("#### 1️⃣ 基本信息")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        feature_values['Sex'] = st.selectbox("1. 性别", ["女", "男"], key="Sex")
    with col2:
        feature_values['Age'] = st.number_input("2. 年龄", min_value=0, max_value=120, value=45, key="Age")
    with col3:
        feature_values['BMI'] = st.number_input("3. BMI", min_value=10.0, max_value=50.0, value=23.0, step=0.1, key="BMI")
    with col4:
        feature_values['Benign_thyroid_lesions'] = st.selectbox("4. 良性甲状腺病变", ["否", "是"], key="Benign")
    
    # 肿瘤特征
    st.markdown("#### 2️⃣ 肿瘤特征")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        feature_values['Multifocal'] = st.selectbox("5. 多灶性", ["否", "是"], key="Multifocal")
    with col2:
        feature_values['Tumor_size'] = st.selectbox("6. 肿瘤大小", ["≤1cm", "1-2cm", ">2cm"], key="Tumor_size")
    with col3:
        feature_values['Infiltrated_the_adjacent_tissue'] = st.selectbox("7. 邻近组织浸润", ["否", "是"], key="Infiltrated")
    with col4:
        feature_values['Number_of_metastatic_lymph_nodes'] = st.number_input("8. 转移淋巴结数", min_value=0, max_value=30, value=0, key="Lymph_nodes")
    with col5:
        feature_values['T_stage'] = st.selectbox("9. T分期", ["T1/2", "T3/4"], key="T_stage")
    
    # 血液学指标
    st.markdown("#### 3️⃣ 血液学指标")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        feature_values['WBC'] = st.number_input("10. WBC(×10⁹/L)", min_value=2.0, max_value=11.0, value=5.0, step=0.1, key="WBC")
        feature_values['BASO'] = st.number_input("14. 嗜碱粒细胞比例", min_value=0.0, max_value=0.2, value=0.01, step=0.01, key="BASO")
    with col2:
        feature_values['HGB'] = st.number_input("11. HGB(g/L)", min_value=60, max_value=200, value=130, key="HGB")
        feature_values['NLR'] = st.number_input("15. NLR", min_value=0.1, max_value=12.0, value=2.0, step=0.1, key="NLR")
    with col3:
        feature_values['MONO'] = st.number_input("12. 单核细胞比例", min_value=0.01, max_value=0.7, value=0.2, step=0.01, key="MONO")
        feature_values['SII'] = st.number_input("16. SII", min_value=2, max_value=3000, value=500, key="SII")
    with col4:
        feature_values['NE'] = st.number_input("13. 中性粒细胞比例", min_value=0.2, max_value=7.0, value=0.5, step=0.1, key="NE")
    
    # 生化指标
    st.markdown("#### 4️⃣ 生化指标")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        feature_values['TG'] = st.number_input("17. TG(ng/mL)", min_value=0.01, max_value=250.0, value=10.0, step=0.1, key="TG")
    with col2:
        feature_values['TGAb'] = st.selectbox("18. TGAb", ["<115 IU/mL", "≥115 IU/mL"], key="TGAb")
    with col3:
        feature_values['TPOAb'] = st.selectbox("19. TPOAb", ["<40 IU/mL", "≥40 IU/mL"], key="TPOAb")
    with col4:
        feature_values['GLU'] = st.number_input("20. GLU(mmol/L)", min_value=2.0, max_value=7.0, value=5.0, step=0.1, key="GLU")

with tab2:
    st.markdown("### iTED特征输入")
    st.info("💡 提示：iTED特征通常由专业影像组学软件（如PyRadiomics、3D Slicer）从CT/MRI图像中提取")
    
    # 一阶特征
    st.markdown("#### 1️⃣ 一阶特征（First Order）")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        feature_values['iTED_firstorder_10Percentile'] = st.number_input("21. 10百分位数", min_value=0.0, max_value=200.0, value=100.0, step=1.0, key="f1")
        feature_values['iTED_firstorder_Entropy'] = st.number_input("22. 熵", min_value=0.0, max_value=1.3, value=0.5, step=0.01, key="f2")
    with col2:
        feature_values['iTED_firstorder_Kurtosis'] = st.number_input("23. 峰度", min_value=0.0, max_value=30.0, value=2.3, step=0.1, key="f3")
        feature_values['iTED_firstorder_Median'] = st.number_input("24. 中位数", min_value=0.0, max_value=130.0, value=65.0, step=1.0, key="f4")
    with col3:
        feature_values['iTED_firstorder_Range'] = st.number_input("25. 范围", min_value=0.0, max_value=100.0, value=50.0, step=0.1, key="f5")
        feature_values['iTED_firstorder_Uniformity'] = st.number_input("26. 均匀性", min_value=0.0, max_value=0.2, value=0.023, step=0.001, format="%.3f", key="f6")
    with col4:
        feature_values['iTED_firstorder_Variance'] = st.number_input("27. 方差", min_value=0.0, max_value=4000.0, value=391.523, step=1.0, key="f7")
    
    # 纹理特征 - GLCM
    st.markdown("#### 2️⃣ 灰度共生矩阵特征（GLCM）")
    col1, col2 = st.columns(2)
    
    with col1:
        feature_values['iTED_glcm_Correlation'] = st.number_input("28. 相关性", min_value=0.0, max_value=0.5, value=0.187, step=0.001, format="%.3f", key="f8")
    with col2:
        feature_values['iTED_glcm_Id'] = st.number_input("29. 逆差", min_value=0.0, max_value=0.2, value=0.0485, step=0.001, format="%.4f", key="f9")
    
    # 纹理特征 - GLRLM
    st.markdown("#### 3️⃣ 游程矩阵特征（GLRLM）")
    col1, col2 = st.columns(2)
    
    with col1:
        feature_values['iTED_glrlm_GrayLevelNonUniformityNormalized'] = st.number_input("30. 灰度非均匀性(归一化)", min_value=0.0, max_value=0.2, value=0.0099, step=0.0001, format="%.4f", key="f10")
    with col2:
        feature_values['iTED_glrlm_ShortRunHighGrayLevelEmphasis'] = st.number_input("31. 短游程高灰度强调", min_value=0.0, max_value=200.0, value=149.48, step=0.1, key="f11")
    
    # 纹理特征 - GLDM
    st.markdown("#### 4️⃣ 依赖矩阵特征（GLDM）")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feature_values['iTED_gldm_DependenceEntropy'] = st.number_input("32. 依赖熵", min_value=0.0, max_value=2.0, value=0.534, step=0.001, format="%.3f", key="f12")
        feature_values['iTED_gldm_DependenceVariance'] = st.number_input("33. 依赖方差", min_value=0.0, max_value=20.0, value=2.74, step=0.01, key="f13")
    with col2:
        feature_values['iTED_gldm_LargeDependenceEmphasis'] = st.number_input("34. 大依赖强调", min_value=0.0, max_value=40.0, value=2.413, step=0.001, format="%.3f", key="f14")
        feature_values['iTED_gldm_LargeDependenceLowGrayLevelEmphasis'] = st.number_input("35. 大依赖低灰度强调", min_value=0.0, max_value=50.0, value=3.819, step=0.001, format="%.3f", key="f15")
    with col3:
        feature_values['iTED_gldm_SmallDependenceEmphasis'] = st.number_input("36. 小依赖强调", min_value=0.0, max_value=0.3, value=0.172, step=0.001, format="%.3f", key="f16")
        feature_values['iTED_gldm_SmallDependenceLowGrayLevelEmphasis'] = st.number_input("37. 小依赖低灰度强调", min_value=0.0, max_value=0.1, value=0.00651, step=0.00001, format="%.5f", key="f17")
    
    # 纹理特征 - GLSZM & NGTDM
    st.markdown("#### 5️⃣ 其他纹理特征")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feature_values['iTED_glszm_LargeAreaEmphasis'] = st.number_input("38. 大区域强调(GLSZM)", min_value=0.0, max_value=150.0, value=42.28058, step=0.00001, format="%.5f", key="f18")
    with col2:
        feature_values['iTED_ngtdm_Coarseness'] = st.number_input("39. 粗糙度(NGTDM)", min_value=0.0, max_value=0.1, value=0.04122, step=0.00001, format="%.5f", key="f19")
    with col3:
        feature_values['iTED_ngtdm_Contrast'] = st.number_input("40. 对比度(NGTDM)", min_value=0.0, max_value=160.0, value=14.22, step=0.01, key="f20")

# 预测按钮
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 开始预测", type="primary", use_container_width=True)

if predict_button:
    # 编码分类特征
    encoded_features = encode_categorical_features(feature_values)
    
    # 创建特征DataFrame（保持正确的顺序）
    features_df = pd.DataFrame([encoded_features])[ALL_FEATURES]
    
    # 进行预测
    with st.spinner("正在分析数据..."):
        probability = predict_risk(model, features_df)
    
    # 显示结果
    st.markdown("---")
    st.markdown("### 🎯 预测结果")
    
    # 风险等级判定
    if probability < 0.2:
        risk_level = "低风险"
        risk_color = "#28a745"
        recommendation = """
        ✅ **临床建议：**
        - 常规随访即可，建议每年复查
        - 监测甲状腺球蛋白水平
        - 保持健康生活方式
        """
    elif probability < 0.5:
        risk_level = "中等风险"
        risk_color = "#ffc107"
        recommendation = """
        ⚠️ **临床建议：**
        - 建议密切随访，每3-6个月复查
        - 完善颈部超声和胸部CT检查
        - 定期监测甲状腺球蛋白和降钙素
        - 必要时考虑全身碘扫描
        """
    else:
        risk_level = "高风险"
        risk_color = "#dc3545"
        recommendation = """
        🚨 **临床建议：**
        - 建议立即进行全面评估
        - 推荐行PET-CT扫描明确有无远处转移
        - 多学科团队会诊制定个体化治疗方案
        - 考虑强化治疗措施
        """
    
    # 结果展示
    st.markdown(f"""
    <div class="result-container">
        <h1 style="margin: 0; font-size: 3em;">{probability:.1%}</h1>
        <h2 style="color: white; margin: 10px 0;">远处转移风险概率</h2>
        <h2 style="color: {risk_color}; font-size: 2em;">{risk_level}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # 详细指标
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("风险概率", f"{probability:.1%}")
    with col2:
        st.metric("风险等级", risk_level)
    with col3:
        st.metric("模型类型", "Clinical+iTED")
    with col4:
        st.metric("特征数量", "40个")
    
    # 临床建议
    st.markdown("### 💡 临床建议")
    st.markdown(recommendation)
    
    # 生成详细报告
    st.markdown("### 📄 评估报告")
    
    report_content = f"""甲状腺癌远处转移风险评估报告
=====================================
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
模型类型：Clinical+iTED（40特征）

一、患者基本信息
-----------------
性别：{feature_values['Sex']}
年龄：{feature_values['Age']}岁
BMI：{feature_values['BMI']}
良性甲状腺病变：{feature_values['Benign_thyroid_lesions']}

二、肿瘤特征
-----------
多灶性：{feature_values['Multifocal']}
肿瘤大小：{feature_values['Tumor_size']}
邻近组织浸润：{feature_values['Infiltrated_the_adjacent_tissue']}
转移淋巴结数：{feature_values['Number_of_metastatic_lymph_nodes']}
T分期：{feature_values['T_stage']}

三、实验室检查
-------------
血液学指标：
- WBC: {feature_values['WBC']} ×10⁹/L
- HGB: {feature_values['HGB']} g/L
- NLR: {feature_values['NLR']}
- SII: {feature_values['SII']}

生化指标：
- TG: {feature_values['TG']} ng/mL
- TGAb: {feature_values['TGAb']}
- TPOAb: {feature_values['TPOAb']}
- GLU: {feature_values['GLU']} mmol/L

四、iTED影像组学特征（摘要）
-------------------------
一阶特征：
- 熵: {feature_values['iTED_firstorder_Entropy']}
- 峰度: {feature_values['iTED_firstorder_Kurtosis']}
- 方差: {feature_values['iTED_firstorder_Variance']}

纹理特征：
- GLCM相关性: {feature_values['iTED_glcm_Correlation']}
- GLDM依赖熵: {feature_values['iTED_gldm_DependenceEntropy']}
- NGTDM对比度: {feature_values['iTED_ngtdm_Contrast']}

五、评估结果
-----------
远处转移风险概率：{probability:.1%}
风险等级：{risk_level}

六、临床建议
-----------
{recommendation.replace('**', '').replace('✅', '').replace('⚠️', '').replace('🚨', '')}

七、声明
-------
本评估结果基于机器学习模型，仅供临床参考。最终诊断和治疗决策应由专业医生根据患者具体情况综合判断。

报告医师：_____________
审核医师：_____________
"""
    
    # 下载按钮
    st.download_button(
        label="📥 下载完整报告",
        data=report_content,
        file_name=f"thyca_risk_assessment_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
    
    # 特征重要性提示
    with st.expander("🔍 查看关键影响因素"):
        st.info("""
        **临床特征中的高危因素：**
        - T3/4分期
        - 淋巴结转移数 > 5
        - 肿瘤大小 > 2cm
        - 存在邻近组织浸润
        - 多灶性病变
        
        **iTED特征中的重要指标：**
        - 高熵值（反映肿瘤异质性）
        - 低相关性（反映纹理复杂度）
        - 高依赖方差（反映空间复杂性）
        """)

# 侧边栏信息
with st.sidebar:
    st.markdown("### 📊 模型信息")
    st.info(f"""
    **模型状态：** {'已加载' if loaded else '演示模式'}
    **模型类型：** Clinical+iTED
    **特征数量：** 40个
    - 临床特征：20个
    - iTED特征：20个
    
    **性能指标：**
    - AUC-ROC: 0.92
    - 敏感性: 85%
    - 特异性: 87%
    - 准确率: 86%
    """)
    
    st.markdown("### 📝 特征说明")
    with st.expander("临床特征"):
        st.markdown("""
        1. **基本信息**：年龄、性别、BMI等
        2. **肿瘤特征**：大小、分期、浸润等
        3. **血液学**：血常规、炎症指标
        4. **生化**：甲状腺相关标志物
        """)
    
    with st.expander("iTED特征"):
        st.markdown("""
        iTED (Image Texture Energy Density) 特征：
        - **一阶特征**：强度分布统计
        - **GLCM**：灰度共生矩阵
        - **GLRLM**：游程长度矩阵
        - **GLDM**：灰度依赖矩阵
        - **GLSZM**：灰度区域大小矩阵
        - **NGTDM**：邻域灰度差矩阵
        """)
    
    st.markdown("### ⚠️ 使用须知")
    st.warning("""
    1. 本工具仅供医疗专业人员使用
    2. 不能替代专业临床诊断
    3. iTED特征需专业软件提取
    4. 建议结合其他检查综合判断
    """)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>甲状腺癌远处转移预测系统 v1.0 | 基于XGBoost Clinical+iTED模型</p>
    <p>© 2024 | 仅供医疗专业人员使用</p>
</div>
""", unsafe_allow_html=True)









