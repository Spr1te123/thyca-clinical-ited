"""
Thyroid Cancer Distant Metastasis Prediction System - Complete Version
Based on Clinical+iTED Model (40 features)
Supports manual input of all feature values
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from datetime import datetime
import json

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

# Debug: Check what files are in the deployment
import os
if st.sidebar.checkbox("Show debug info", value=False):
    st.sidebar.write("Current directory:", os.getcwd())
    st.sidebar.write("Files in current directory:", os.listdir('.'))
    st.sidebar.write("App file location:", os.path.abspath(__file__))

# Define all 40 features configuration (in the actual order used during model training)
ALL_FEATURES = [
    # Clinical features (20) - Note: Sex comes before Age!
    'Sex', 'Age', 'BMI', 'Benign_thyroid_lesions', 'Multifocal', 
    'Tumor_size', 'Infiltrated_the_adjacent_tissue', 'Number_of_metastatic_lymph_nodes',
    'T_stage', 'WBC', 'HGB', 'MONO', 'NE', 'BASO', 'NLR', 'SII', 
    'TG', 'TGAb', 'TPOAb', 'GLU',
    
    # iTED features (20)
    'iTED_firstorder_10Percentile', 'iTED_firstorder_Entropy', 'iTED_firstorder_Kurtosis',
    'iTED_firstorder_Median', 'iTED_firstorder_Range', 'iTED_firstorder_Uniformity',
    'iTED_firstorder_Variance', 'iTED_glcm_Correlation', 'iTED_glcm_Id',
    'iTED_glrlm_GrayLevelNonUniformityNormalized', 'iTED_glrlm_ShortRunHighGrayLevelEmphasis',
    'iTED_gldm_DependenceEntropy', 'iTED_gldm_DependenceVariance',
    'iTED_gldm_LargeDependenceEmphasis', 'iTED_gldm_LargeDependenceLowGrayLevelEmphasis',
    'iTED_gldm_SmallDependenceEmphasis', 'iTED_gldm_SmallDependenceLowGrayLevelEmphasis',
    'iTED_glszm_LargeAreaEmphasis', 'iTED_ngtdm_Coarseness', 'iTED_ngtdm_Contrast'
]

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_loaded = False


@st.cache_resource
def load_model():
    """Load XGBoost model"""
    import os
    
    # 尝试多个可能的路径
    possible_paths = [
        'models/best_model_Clinical_iTED.json',
        './models/best_model_Clinical_iTED.json',
        '/mount/src/thyca-clinical-ited/models/best_model_Clinical_iTED.json',  # Streamlit Cloud 特定路径
        os.path.join(os.getcwd(), 'models', 'best_model_Clinical_iTED.json'),
    ]
    
    for path in possible_paths:
        st.sidebar.write(f"尝试路径: {path}")
        st.sidebar.write(f"存在: {os.path.exists(path)}")
        
        if os.path.exists(path):
            try:
                model = xgb.XGBClassifier()
                model.load_model(path)
                st.sidebar.success(f"从 {path} 加载成功!")
                return model, True
            except Exception as e:
                st.sidebar.error(f"加载失败 {path}: {e}")
    
    return None, False

def encode_categorical_features(features_dict):
    """Encode categorical features"""
    encoded = features_dict.copy()
    
    # Sex encoding
    if 'Sex' in encoded:
        encoded['Sex'] = 1 if encoded['Sex'] == 'Male' else 0
    
    # Benign lesions encoding
    if 'Benign_thyroid_lesions' in encoded:
        encoded['Benign_thyroid_lesions'] = 1 if encoded['Benign_thyroid_lesions'] == 'Yes' else 0
    
    # Multifocal encoding
    if 'Multifocal' in encoded:
        encoded['Multifocal'] = 1 if encoded['Multifocal'] == 'Yes' else 0
    
    # Tumor size encoding
    if 'Tumor_size' in encoded:
        size_map = {'≤1cm': 0, '1-2cm': 1, '>2cm': 2}
        encoded['Tumor_size'] = size_map.get(encoded['Tumor_size'], 0)
    
    # Adjacent tissue infiltration encoding
    if 'Infiltrated_the_adjacent_tissue' in encoded:
        encoded['Infiltrated_the_adjacent_tissue'] = 1 if encoded['Infiltrated_the_adjacent_tissue'] == 'Yes' else 0
    
    # T stage encoding
    if 'T_stage' in encoded:
        encoded['T_stage'] = 1 if encoded['T_stage'] == 'T3/4' else 0
    
    # Antibody encoding
    if 'TGAb' in encoded:
        encoded['TGAb'] = 1 if encoded['TGAb'] == '≥115 IU/mL' else 0
    
    if 'TPOAb' in encoded:
        encoded['TPOAb'] = 1 if encoded['TPOAb'] == '≥40 IU/mL' else 0
    
    return encoded

def predict_risk(model, features_df):
    """Predict risk using the model"""
    if model is not None:
        # Use real model for prediction
        probability = model.predict_proba(features_df)[0][1]
    else:
        # Demo mode - calculate simulated probability based on features
        features = features_df.iloc[0]
        risk_score = 0.1  # Base risk
        
        # Clinical feature weights
        if features.get('Sex', 0) == 1:  # Male
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
        
        # iTED feature impact (simulated)
        if features.get('iTED_firstorder_Entropy', 0) > 0.8:
            risk_score += 0.10
        if features.get('iTED_glcm_Correlation', 0) < 0.1:
            risk_score += 0.08
        
        probability = min(risk_score, 0.95)
    
    return probability

# Main interface
st.title("🏥 Thyroid Cancer Distant Metastasis Prediction System")
st.markdown("### Comprehensive Assessment Based on Clinical+iTED Model (40 Features)")

# Load model
model, loaded = load_model()
if loaded:
    st.success("✅ Model loaded successfully")
else:
    st.warning("⚠️ Using demo mode (model file not found)")

# Instructions
with st.expander("📋 Instructions", expanded=False):
    st.info("""
    **Input Requirements:**
    1. Clinical features (20): Patient demographics, tumor characteristics, blood and biochemical markers
    2. iTED features (20): Texture features extracted from medical imaging
    
    **Notes:**
    - Ensure all values are accurate
    - iTED features are typically calculated by professional software
    - Consult relevant departments if uncertain about any value
    
    **Model Performance:**
    - AUC-ROC: 0.92
    - Sensitivity: 85%
    - Specificity: 87%
    """)

# Create input form
st.markdown("---")
feature_values = {}

# Use tabs to organize inputs
tab1, tab2 = st.tabs(["📝 Clinical Features (1-20)", "🔬 iTED Features (21-40)"])

with tab1:
    st.markdown("### Clinical Feature Input")
    
    # Basic information
    st.markdown("#### 1️⃣ Basic Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        feature_values['Sex'] = st.selectbox("1. Sex", ["Female", "Male"], key="Sex")
    with col2:
        feature_values['Age'] = st.number_input("2. Age", min_value=0, max_value=120, value=45, key="Age")
    with col3:
        feature_values['BMI'] = st.number_input("3. BMI", min_value=10.0, max_value=50.0, value=23.0, step=0.1, key="BMI")
    with col4:
        feature_values['Benign_thyroid_lesions'] = st.selectbox("4. Benign thyroid lesions", ["No", "Yes"], key="Benign")
    
    # Tumor characteristics
    st.markdown("#### 2️⃣ Tumor Characteristics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        feature_values['Multifocal'] = st.selectbox("5. Multifocal", ["No", "Yes"], key="Multifocal")
    with col2:
        feature_values['Tumor_size'] = st.selectbox("6. Tumor size", ["≤1cm", "1-2cm", ">2cm"], key="Tumor_size")
    with col3:
        feature_values['Infiltrated_the_adjacent_tissue'] = st.selectbox("7. Adjacent tissue infiltration", ["No", "Yes"], key="Infiltrated")
    with col4:
        feature_values['Number_of_metastatic_lymph_nodes'] = st.number_input("8. Metastatic lymph nodes", min_value=0, max_value=30, value=0, key="Lymph_nodes")
    with col5:
        feature_values['T_stage'] = st.selectbox("9. T stage", ["T1/2", "T3/4"], key="T_stage")
    
    # Blood markers
    st.markdown("#### 3️⃣ Blood Markers")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        feature_values['WBC'] = st.number_input("10. WBC(×10⁹/L)", min_value=2.0, max_value=11.0, value=5.0, step=0.1, key="WBC")
        feature_values['BASO'] = st.number_input("14. Basophil ratio", min_value=0.0, max_value=0.2, value=0.01, step=0.01, key="BASO")
    with col2:
        feature_values['HGB'] = st.number_input("11. HGB(g/L)", min_value=60, max_value=200, value=130, key="HGB")
        feature_values['NLR'] = st.number_input("15. NLR", min_value=0.1, max_value=12.0, value=2.0, step=0.1, key="NLR")
    with col3:
        feature_values['MONO'] = st.number_input("12. Monocyte ratio", min_value=0.01, max_value=0.7, value=0.2, step=0.01, key="MONO")
        feature_values['SII'] = st.number_input("16. SII", min_value=2, max_value=3000, value=500, key="SII")
    with col4:
        feature_values['NE'] = st.number_input("13. Neutrophil ratio", min_value=0.2, max_value=7.0, value=0.5, step=0.1, key="NE")
    
    # Biochemical markers
    st.markdown("#### 4️⃣ Biochemical Markers")
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
    st.markdown("### iTED Feature Input")
    st.info("💡 Tip: iTED features are typically extracted from CT/MRI images using professional radiomics software (e.g., PyRadiomics, 3D Slicer)")
    
    # First-order features
    st.markdown("#### 1️⃣ First-Order Features")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        feature_values['iTED_firstorder_10Percentile'] = st.number_input("21. 10th Percentile", min_value=0.0, max_value=200.0, value=100.0, step=1.0, key="f1")
        feature_values['iTED_firstorder_Entropy'] = st.number_input("22. Entropy", min_value=0.0, max_value=1.3, value=0.5, step=0.01, key="f2")
    with col2:
        feature_values['iTED_firstorder_Kurtosis'] = st.number_input("23. Kurtosis", min_value=0.0, max_value=30.0, value=2.3, step=0.1, key="f3")
        feature_values['iTED_firstorder_Median'] = st.number_input("24. Median", min_value=0.0, max_value=130.0, value=65.0, step=1.0, key="f4")
    with col3:
        feature_values['iTED_firstorder_Range'] = st.number_input("25. Range", min_value=0.0, max_value=100.0, value=50.0, step=0.1, key="f5")
        feature_values['iTED_firstorder_Uniformity'] = st.number_input("26. Uniformity", min_value=0.0, max_value=0.2, value=0.023, step=0.001, format="%.3f", key="f6")
    with col4:
        feature_values['iTED_firstorder_Variance'] = st.number_input("27. Variance", min_value=0.0, max_value=4000.0, value=391.523, step=1.0, key="f7")
    
    # Texture features - GLCM
    st.markdown("#### 2️⃣ Gray Level Co-occurrence Matrix (GLCM)")
    col1, col2 = st.columns(2)
    
    with col1:
        feature_values['iTED_glcm_Correlation'] = st.number_input("28. Correlation", min_value=0.0, max_value=0.5, value=0.187, step=0.001, format="%.3f", key="f8")
    with col2:
        feature_values['iTED_glcm_Id'] = st.number_input("29. Inverse Difference", min_value=0.0, max_value=0.2, value=0.0485, step=0.001, format="%.4f", key="f9")
    
    # Texture features - GLRLM
    st.markdown("#### 3️⃣ Gray Level Run Length Matrix (GLRLM)")
    col1, col2 = st.columns(2)
    
    with col1:
        feature_values['iTED_glrlm_GrayLevelNonUniformityNormalized'] = st.number_input("30. Gray Level Non-Uniformity (Normalized)", min_value=0.0, max_value=0.2, value=0.0099, step=0.0001, format="%.4f", key="f10")
    with col2:
        feature_values['iTED_glrlm_ShortRunHighGrayLevelEmphasis'] = st.number_input("31. Short Run High Gray Level Emphasis", min_value=0.0, max_value=200.0, value=149.48, step=0.1, key="f11")
    
    # Texture features - GLDM
    st.markdown("#### 4️⃣ Gray Level Dependence Matrix (GLDM)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feature_values['iTED_gldm_DependenceEntropy'] = st.number_input("32. Dependence Entropy", min_value=0.0, max_value=2.0, value=0.534, step=0.001, format="%.3f", key="f12")
        feature_values['iTED_gldm_DependenceVariance'] = st.number_input("33. Dependence Variance", min_value=0.0, max_value=20.0, value=2.74, step=0.01, key="f13")
    with col2:
        feature_values['iTED_gldm_LargeDependenceEmphasis'] = st.number_input("34. Large Dependence Emphasis", min_value=0.0, max_value=40.0, value=2.413, step=0.001, format="%.3f", key="f14")
        feature_values['iTED_gldm_LargeDependenceLowGrayLevelEmphasis'] = st.number_input("35. Large Dependence Low Gray Level Emphasis", min_value=0.0, max_value=50.0, value=3.819, step=0.001, format="%.3f", key="f15")
    with col3:
        feature_values['iTED_gldm_SmallDependenceEmphasis'] = st.number_input("36. Small Dependence Emphasis", min_value=0.0, max_value=0.3, value=0.172, step=0.001, format="%.3f", key="f16")
        feature_values['iTED_gldm_SmallDependenceLowGrayLevelEmphasis'] = st.number_input("37. Small Dependence Low Gray Level Emphasis", min_value=0.0, max_value=0.1, value=0.00651, step=0.00001, format="%.5f", key="f17")
    
    # Texture features - GLSZM & NGTDM
    st.markdown("#### 5️⃣ Other Texture Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feature_values['iTED_glszm_LargeAreaEmphasis'] = st.number_input("38. Large Area Emphasis (GLSZM)", min_value=0.0, max_value=150.0, value=42.28058, step=0.00001, format="%.5f", key="f18")
    with col2:
        feature_values['iTED_ngtdm_Coarseness'] = st.number_input("39. Coarseness (NGTDM)", min_value=0.0, max_value=0.1, value=0.04122, step=0.00001, format="%.5f", key="f19")
    with col3:
        feature_values['iTED_ngtdm_Contrast'] = st.number_input("40. Contrast (NGTDM)", min_value=0.0, max_value=160.0, value=14.22, step=0.01, key="f20")

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
    
    # Risk level determination
    if probability < 0.2:
        risk_level = "Low Risk"
        risk_color = "#28a745"
        recommendation = """
        ✅ **Clinical Recommendations:**
        - Routine follow-up is sufficient, annual review recommended
        - Monitor thyroglobulin levels
        - Maintain healthy lifestyle
        """
    elif probability < 0.5:
        risk_level = "Moderate Risk"
        risk_color = "#ffc107"
        recommendation = """
        ⚠️ **Clinical Recommendations:**
        - Close follow-up recommended, review every 3-6 months
        - Complete neck ultrasound and chest CT examination
        - Regular monitoring of thyroglobulin and calcitonin
        - Consider whole-body iodine scan if necessary
        """
    else:
        risk_level = "High Risk"
        risk_color = "#dc3545"
        recommendation = """
        🚨 **Clinical Recommendations:**
        - Immediate comprehensive evaluation recommended
        - PET-CT scan recommended to detect distant metastasis
        - Multidisciplinary team consultation for personalized treatment plan
        - Consider intensive treatment measures
        """
    
    # Results display
    st.markdown(f"""
    <div class="result-container">
        <h1 style="margin: 0; font-size: 3em;">{probability:.1%}</h1>
        <h2 style="color: white; margin: 10px 0;">Distant Metastasis Risk Probability</h2>
        <h2 style="color: {risk_color}; font-size: 2em;">{risk_level}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Risk Probability", f"{probability:.1%}")
    with col2:
        st.metric("Risk Level", risk_level)
    with col3:
        st.metric("Model Type", "Clinical+iTED")
    with col4:
        st.metric("Features", "40")
    
    # Clinical recommendations
    st.markdown("### 💡 Clinical Recommendations")
    st.markdown(recommendation)
    
    # Generate detailed report
    st.markdown("### 📄 Assessment Report")
    
    report_content = f"""Thyroid Cancer Distant Metastasis Risk Assessment Report
=====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Type: Clinical+iTED (40 features)

I. Patient Demographics
-----------------
Sex: {feature_values['Sex']}
Age: {feature_values['Age']} years
BMI: {feature_values['BMI']}
Benign thyroid lesions: {feature_values['Benign_thyroid_lesions']}

II. Tumor Characteristics
-----------
Multifocal: {feature_values['Multifocal']}
Tumor size: {feature_values['Tumor_size']}
Adjacent tissue infiltration: {feature_values['Infiltrated_the_adjacent_tissue']}
Metastatic lymph nodes: {feature_values['Number_of_metastatic_lymph_nodes']}
T stage: {feature_values['T_stage']}

III. Laboratory Tests
-------------
Blood markers:
- WBC: {feature_values['WBC']} ×10⁹/L
- HGB: {feature_values['HGB']} g/L
- NLR: {feature_values['NLR']}
- SII: {feature_values['SII']}

Biochemical markers:
- TG: {feature_values['TG']} ng/mL
- TGAb: {feature_values['TGAb']}
- TPOAb: {feature_values['TPOAb']}
- GLU: {feature_values['GLU']} mmol/L

IV. iTED Radiomics Features (Summary)
-------------------------
First-order features:
- Entropy: {feature_values['iTED_firstorder_Entropy']}
- Kurtosis: {feature_values['iTED_firstorder_Kurtosis']}
- Variance: {feature_values['iTED_firstorder_Variance']}

Texture features:
- GLCM Correlation: {feature_values['iTED_glcm_Correlation']}
- GLDM Dependence Entropy: {feature_values['iTED_gldm_DependenceEntropy']}
- NGTDM Contrast: {feature_values['iTED_ngtdm_Contrast']}

V. Assessment Results
-----------
Distant metastasis risk probability: {probability:.1%}
Risk level: {risk_level}

VI. Clinical Recommendations
-----------
{recommendation.replace('**', '').replace('✅', '').replace('⚠️', '').replace('🚨', '')}

VII. Disclaimer
-------
This assessment is based on machine learning models and is for clinical reference only. Final diagnosis and treatment decisions should be made by professional physicians based on comprehensive patient evaluation.

Reporting Physician: _____________
Reviewing Physician: _____________
"""
    
    # Download button
    st.download_button(
        label="📥 Download Full Report",
        data=report_content,
        file_name=f"thyca_risk_assessment_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
    
    # Feature importance tips
    with st.expander("🔍 View Key Risk Factors"):
        st.info("""
        **High-risk clinical features:**
        - T3/4 stage
        - Lymph node metastases > 5
        - Tumor size > 2cm
        - Adjacent tissue infiltration present
        - Multifocal disease
        
        **Important iTED features:**
        - High entropy (reflects tumor heterogeneity)
        - Low correlation (reflects texture complexity)
        - High dependence variance (reflects spatial complexity)
        """)

# Sidebar information
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.info(f"""
    **Model Status:** {'Loaded' if loaded else 'Demo Mode'}
    **Model Type:** Clinical+iTED
    **Features:** 40
    - Clinical: 20
    - iTED: 20
    
    **Performance:**
    - AUC-ROC: 0.92
    - Sensitivity: 85%
    - Specificity: 87%
    - Accuracy: 86%
    """)
    
    st.markdown("### 📝 Feature Description")
    with st.expander("Clinical Features"):
        st.markdown("""
        1. **Demographics**: Age, sex, BMI
        2. **Tumor**: Size, stage, infiltration
        3. **Blood**: Complete blood count, inflammatory markers
        4. **Biochemistry**: Thyroid-related markers
        """)
    
    with st.expander("iTED Features"):
        st.markdown("""
        iTED (Image Texture Energy Density) features:
        - **First-order**: Intensity distribution statistics
        - **GLCM**: Gray-level co-occurrence matrix
        - **GLRLM**: Gray-level run length matrix
        - **GLDM**: Gray-level dependence matrix
        - **GLSZM**: Gray-level size zone matrix
        - **NGTDM**: Neighborhood gray-tone difference matrix
        """)
    
    st.markdown("### ⚠️ Usage Notice")
    st.warning("""
    1. For medical professionals only
    2. Cannot replace professional clinical diagnosis
    3. iTED features require professional extraction
    4. Recommend comprehensive evaluation with other tests
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Thyroid Cancer Distant Metastasis Prediction System v1.0 | Based on XGBoost Clinical+iTED Model</p>
    <p>© 2024 | For Medical Professionals Only</p>
</div>
""", unsafe_allow_html=True)









