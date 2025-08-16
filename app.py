"""
甲状腺癌远处转移预测系统
基于LightGBM模型（7特征）with SHAP分析
"""
import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import base64
import tempfile
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import sys

# 禁用matplotlib的交互模式
plt.ioff()

# Page configuration
st.set_page_config(
    page_title="甲状腺癌远处转移预测系统",
    page_icon="🏥",
    layout="wide"
)

# ========== 模型配置部分 ==========

# 模型Base64字符串（请将model_base64.txt的内容粘贴到这里）
MODEL_BASE64 = ""  # 请粘贴您的base64字符串

# 如果不想硬编码，可以从文件读取
try:
    with open('model_base64.txt', 'r') as f:
        MODEL_BASE64 = f.read().strip()
except:
    pass

# 模型特征配置
MODEL_CONFIG = {
    "model_name": "All_Features",
    "features": [
        "Multifocal",
        "T_stage",
        "glcm_JointEntropy",
        "glrlm_GrayLevelNonUniformityNormalized",
        "shape_SurfaceArea",
        "iTED_firstorder_Energy",
        "iTED_firstorder_Variance"
    ],
    "feature_types": {
        "Multifocal": "clinical",
        "T_stage": "clinical",
        "glcm_JointEntropy": "radiomics",
        "glrlm_GrayLevelNonUniformityNormalized": "radiomics",
        "shape_SurfaceArea": "radiomics",
        "iTED_firstorder_Energy": "iTED",
        "iTED_firstorder_Variance": "iTED"
    },
    "feature_counts": {
        "clinical": 2,
        "radiomics": 3,
        "iTED": 2
    }
}

# 特征编码映射
ENCODING_MAPPINGS = {
    "Multifocal": {"No": 0, "Yes": 1},
    "T_stage": {"T1": 0, "T2": 1, "T3": 2, "T4": 3}
}

# 特征范围配置
FEATURE_RANGES = {
    "iTED_firstorder_Energy": {
        "min": 0,
        "max": 4000000,
        "default": 100000,
        "step": 10000.0,
        "format": "%.0f",
        "help": "iTED一阶能量特征 (范围: 0-3,740,000)"
    },
    "iTED_firstorder_Variance": {
        "min": 0,
        "max": 3500,
        "default": 100,
        "step": 10.0,
        "format": "%.1f",
        "help": "iTED一阶方差特征 (范围: 0-3,339)"
    },
    "glcm_JointEntropy": {
        "min": 0,
        "max": 10,
        "default": 5,
        "step": 0.1,
        "format": "%.2f",
        "help": "灰度共生矩阵联合熵 (范围: 1.4-8.8)"
    },
    "glrlm_GrayLevelNonUniformityNormalized": {
        "min": 0,
        "max": 0.5,
        "default": 0.2,
        "step": 0.01,
        "format": "%.3f",
        "help": "归一化灰度游程矩阵非均匀性 (范围: 0.04-0.45)"
    },
    "shape_SurfaceArea": {
        "min": 0,
        "max": 30000,
        "default": 1000,
        "step": 100.0,
        "format": "%.1f",
        "help": "肿瘤表面积 (范围: 35-29,000)"
    }
}

# 模型性能指标
MODEL_PERFORMANCE = {
    "internal": {
        "AUC-ROC": 0.6724,
        "AUC-ROC_CI": [0.5852, 0.7605],
        "AUC-PR": 0.2026,
        "Sensitivity": 0.7500,
        "Specificity": 0.5948,
        "NPV": 0.9381,
        "PPV": 0.2250,
        "F3-Score": 0.6081,
        "MCC": 0.2372,
        "Brier_Score": 0.1123
    },
    "external": {
        "AUC-ROC": 0.7538,
        "AUC-ROC_CI": [0.6417, 0.8606],
        "AUC-PR": 0.2127,
        "Sensitivity": 0.8000,
        "Specificity": 0.7077,
        "NPV": 0.9684,
        "PPV": 0.2400,
        "F3-Score": 0.6486,
        "MCC": 0.3253,
        "Brier_Score": 0.0832
    }
}

# 最优阈值
OPTIMAL_THRESHOLD = 0.10

# ========== 样式设置 ==========
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

# ========== 函数定义 ==========

def load_model():
    """加载LightGBM模型"""
    if not MODEL_BASE64:
        return None, False

    try:
        # 解码base64
        model_bytes = base64.b64decode(MODEL_BASE64)

        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            f.write(model_bytes)
            temp_path = f.name

        # 加载模型
        booster = lgb.Booster(model_file=temp_path)

        # 删除临时文件
        os.unlink(temp_path)

        return booster, True
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None, False

def encode_categorical_features(features_dict):
    """编码分类特征"""
    encoded = {}

    for feature, value in features_dict.items():
        if feature in ENCODING_MAPPINGS:
            if value in ENCODING_MAPPINGS[feature]:
                encoded[feature] = ENCODING_MAPPINGS[feature][value]
            else:
                st.error(f"未知的{feature}值: {value}")
                return None
        else:
            encoded[feature] = value

    return encoded

def predict_risk(booster, features_df):
    """使用模型预测"""
    if booster is not None:
        try:
            probability = booster.predict(features_df, num_iteration=booster.best_iteration)[0]
            return probability
        except Exception as e:
            st.error(f"预测失败: {str(e)}")
            return None
    else:
        # 演示模式 - 简单的规则计算
        risk_score = 0.135  # 基线风险 (13.5% M1患病率)

        # 根据特征值调整风险
        if features_df['Multifocal'].iloc[0] == 1:
            risk_score += 0.10
        if features_df['T_stage'].iloc[0] >= 2:  # T3 or T4
            risk_score += 0.15
        if features_df['shape_SurfaceArea'].iloc[0] > 5000:
            risk_score += 0.08
        if features_df['glcm_JointEntropy'].iloc[0] > 6:
            risk_score += 0.05

        # iTED特征贡献
        iTED_sum = features_df['iTED_firstorder_Energy'].iloc[0] / 1000000 + \
                   features_df['iTED_firstorder_Variance'].iloc[0] / 1000
        risk_score += min(iTED_sum * 0.05, 0.10)

        # 确保概率在0-1之间
        return min(max(risk_score, 0.0), 1.0)

def display_shap_plotly(feature_values, features_df, probability):
    """显示SHAP分析（演示模式）"""
    st.markdown("### 🎯 特征贡献分析（SHAP演示模式）")

    # 创建模拟的特征贡献
    feature_names = []
    shap_values = []

    for i, feature in enumerate(MODEL_CONFIG['features']):
        value = features_df.iloc[0, i]

        # 创建显示名称
        if feature in ENCODING_MAPPINGS:
            for k, v in ENCODING_MAPPINGS[feature].items():
                if v == value:
                    display_name = f"{feature}={k}"
                    break
            else:
                display_name = f"{feature}={value}"
        else:
            display_name = f"{feature}={value:.2f}" if isinstance(value, float) else f"{feature}={value}"

        feature_names.append(display_name)

        # 模拟SHAP值
        if feature == 'Multifocal' and value == 1:
            shap_val = 0.10
        elif feature == 'T_stage' and value >= 2:
            shap_val = 0.15
        elif feature == 'shape_SurfaceArea' and value > 5000:
            shap_val = 0.08
        elif feature == 'glcm_JointEntropy' and value > 6:
            shap_val = 0.05
        elif feature.startswith('iTED_'):
            shap_val = np.random.uniform(-0.05, 0.05)
        else:
            shap_val = np.random.uniform(-0.02, 0.02)

        shap_values.append(shap_val)

    # 创建数据框
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP': shap_values
    })
    shap_df['abs_SHAP'] = abs(shap_df['SHAP'])
    shap_df = shap_df.sort_values('SHAP')

    # 创建条形图
    fig = go.Figure()

    colors = ['#FF0000' if x > 0 else '#0000FF' for x in shap_df['SHAP']]

    fig.add_trace(go.Bar(
        y=shap_df['Feature'],
        x=shap_df['SHAP'],
        orientation='h',
        marker_color=colors,
        text=[f'{v:.3f}' for v in shap_df['SHAP']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>SHAP值: %{x:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title='特征对远处转移风险预测的贡献（SHAP值）',
        xaxis_title='SHAP值（对预测的贡献）',
        yaxis_title='特征',
        height=400,
        showlegend=False,
        xaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            range=[-0.3, 0.3]
        ),
        plot_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)

    # 显示解释
    st.info(f"""
    **图表说明：**
    - 🔴 红色条：增加远处转移风险的特征
    - 🔵 蓝色条：降低远处转移风险的特征
    - 条的长度表示特征对预测的影响程度
    - 基线风险: 13.5% (M1患病率)
    - 您的预测风险: {probability:.1%}
    
    *注意：当前为演示模式，SHAP值为模拟值。加载真实模型以显示准确的SHAP分析。*
    """)

    # 显示主要风险因素
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔴 主要风险增加因素：**")
        risk_features = shap_df[shap_df['SHAP'] > 0].nlargest(3, 'abs_SHAP')
        if len(risk_features) > 0:
            for _, row in risk_features.iterrows():
                st.write(f"• {row['Feature']}: +{row['SHAP']:.3f}")
        else:
            st.write("无风险增加因素")

    with col2:
        st.markdown("**🔵 主要风险降低因素：**")
        protective_features = shap_df[shap_df['SHAP'] < 0].nlargest(3, 'abs_SHAP')
        if len(protective_features) > 0:
            for _, row in protective_features.iterrows():
                st.write(f"• {row['Feature']}: {row['SHAP']:.3f}")
        else:
            st.write("无风险降低因素")

def display_shap_analysis(booster, feature_values, features_df, probability):
    """显示SHAP分析 - 支持真实模型和演示模式"""
    if booster is not None:
        # 真实模型 - 使用真实SHAP分析
        try:
            st.markdown("### 🎯 SHAP特征贡献分析")

            # 创建SHAP解释器
            with st.spinner("计算SHAP值..."):
                explainer = shap.TreeExplainer(booster)
                shap_values = explainer.shap_values(features_df)

                # 获取期望值
                expected_value = explainer.expected_value
                if isinstance(expected_value, list):
                    expected_value = expected_value[0]

            # 创建特征名称
            feature_names = []
            for i, feature in enumerate(MODEL_CONFIG['features']):
                value = features_df.iloc[0, i]
                if feature in ENCODING_MAPPINGS:
                    for k, v in ENCODING_MAPPINGS[feature].items():
                        if v == value:
                            feature_names.append(f"{feature}={k}")
                            break
                    else:
                        feature_names.append(f"{feature}={value}")
                else:
                    feature_names.append(f"{feature}={value:.3f}")

            # 显示SHAP分析结果
            st.subheader("SHAP分析结果")

            # 显示基本信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("基线风险", f"{expected_value:.1%}")
            with col2:
                st.metric("SHAP贡献", f"{shap_values[0].sum():.3f}")
            with col3:
                st.metric("最终预测", f"{probability:.1%}")

            # 创建SHAP数据框
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP': shap_values[0] if len(shap_values.shape) == 1 else shap_values,
                'Feature_Value': features_df.iloc[0].values
            })
            shap_df['abs_SHAP'] = abs(shap_df['SHAP'])
            shap_df = shap_df.sort_values('SHAP')

            # 创建条形图
            fig = go.Figure()

            colors = ['#FF0000' if x > 0 else '#0000FF' for x in shap_df['SHAP']]

            fig.add_trace(go.Bar(
                y=shap_df['Feature'],
                x=shap_df['SHAP'],
                orientation='h',
                marker_color=colors,
                text=[f'{v:.3f}' for v in shap_df['SHAP']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>SHAP值: %{x:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title='特征对远处转移风险预测的贡献（SHAP值）',
                xaxis_title='SHAP值（对预测的贡献）',
                yaxis_title='特征',
                height=400,
                showlegend=False,
                xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
                plot_bgcolor='white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # 添加预测路径分析（保留原始功能）
            st.subheader("📈 预测路径分析")

            # 按SHAP值排序
            top_features = shap_df.sort_values('SHAP', ascending=True)

            # 计算累积效应
            cumulative_effects = [expected_value]
            for shap_val in top_features['SHAP']:
                cumulative_effects.append(cumulative_effects[-1] + shap_val)

            # 创建路径图
            fig_path = go.Figure()

            # 添加基线
            fig_path.add_hline(
                y=expected_value,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"基线风险: {expected_value:.1%} (平均患者)",
                annotation_position="left"
            )

            # 添加每个特征的贡献
            x_positions = list(range(len(top_features) + 1))
            feature_labels = ['基线'] + top_features['Feature'].tolist()

            # 绘制累积效应线
            fig_path.add_trace(go.Scatter(
                x=x_positions,
                y=cumulative_effects,
                mode='lines+markers+text',
                line=dict(color='darkblue', width=3),
                marker=dict(size=10, color='darkblue'),
                text=[f"{val:.1%}" for val in cumulative_effects],
                textposition="top center",
                name='累积预测',
                hovertemplate='%{y:.1%}<extra></extra>'
            ))

            # 为每个特征添加贡献条
            for i, (idx, row) in enumerate(top_features.iterrows()):
                color = 'rgba(255,0,0,0.3)' if row['SHAP'] > 0 else 'rgba(0,0,255,0.3)'
                fig_path.add_shape(
                    type="rect",
                    x0=i+0.8, x1=i+1.2,
                    y0=cumulative_effects[i],
                    y1=cumulative_effects[i+1],
                    fillcolor=color,
                    line=dict(width=0)
                )

                # 添加SHAP值标签
                mid_y = (cumulative_effects[i] + cumulative_effects[i+1]) / 2
                fig_path.add_annotation(
                    x=i+1,
                    y=mid_y,
                    text=f"{row['SHAP']:+.3f}",
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )

            # 添加最终预测线
            fig_path.add_hline(
                y=probability,
                line_dash="solid",
                line_color="darkgreen",
                line_width=2,
                annotation_text=f"最终预测: {probability:.1%}",
                annotation_position="right"
            )

            fig_path.update_layout(
                title="从基线风险到最终预测的累积路径",
                xaxis=dict(
                    tickmode='array',
                    tickvals=x_positions,
                    ticktext=feature_labels,
                    tickangle=-45
                ),
                yaxis=dict(
                    title="预测概率",
                    tickformat='.0%',
                    range=[0, 1]
                ),
                height=500,
                showlegend=False,
                plot_bgcolor='white',
                margin=dict(b=150)
            )

            st.plotly_chart(fig_path, use_container_width=True)

            # 添加路径解释
            st.info("""
            **路径图说明：**
            - 灰色虚线：基线风险（平均患者风险）
            - 蓝色线：显示预测如何随每个特征的贡献而变化
            - 红色矩形：增加风险的特征贡献
            - 蓝色矩形：降低风险的特征贡献
            - 绿色实线：最终预测结果
            
            此图显示模型如何通过每个特征的贡献从基线风险得出您的预测风险。
            """)

            # 显示详细信息
            st.info(f"""
            **SHAP分析总结：**
            - 基线风险（平均患者）: {expected_value:.1%}
            - 总SHAP值: {shap_values[0].sum():.3f}
            - 最终预测概率: {probability:.1%}
            
            *注：这是基于真实模型的SHAP分析*
            """)

            # 显示主要风险因素
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🔴 主要风险增加因素：**")
                risk_features = shap_df[shap_df['SHAP'] > 0].nlargest(3, 'abs_SHAP')
                if len(risk_features) > 0:
                    for _, row in risk_features.iterrows():
                        st.write(f"• {row['Feature']}: +{row['SHAP']:.3f}")
                else:
                    st.write("无风险增加因素")

            with col2:
                st.markdown("**🔵 主要风险降低因素：**")
                protective_features = shap_df[shap_df['SHAP'] < 0].nlargest(3, 'abs_SHAP')
                if len(protective_features) > 0:
                    for _, row in protective_features.iterrows():
                        st.write(f"• {row['Feature']}: {row['SHAP']:.3f}")
                else:
                    st.write("无风险降低因素")

        except Exception as e:
            st.error(f"SHAP分析错误: {str(e)}")
            # 回退到演示模式
            display_shap_plotly(feature_values, features_df, probability)
    else:
        # 演示模式
        display_shap_plotly(feature_values, features_df, probability)

# ========== 主界面 ==========

st.title("🏥 甲状腺癌远处转移预测系统")
st.markdown("### 基于LightGBM多模态融合模型（7特征）with SHAP分析")
st.markdown("*优化用于高敏感性检测远处转移*")

# 显示Python版本信息
st.sidebar.markdown("### 📊 系统信息")
st.sidebar.info(f"Python: {sys.version.split()[0]}")
st.sidebar.info(f"NumPy: {np.__version__}")
st.sidebar.info(f"SHAP: {shap.__version__}")
st.sidebar.info(f"LightGBM: {lgb.__version__}")
st.sidebar.info(f"模型: {MODEL_CONFIG['model_name']}")

# 加载模型
model, loaded = load_model()
if loaded:
    st.success("✅ 模型从Base64字符串成功加载")
else:
    st.warning("⚠️ 使用演示模式 - SHAP分析将显示模拟值")
    with st.expander("💡 如何加载真实模型？"):
        st.markdown("""
        要使用真实的LightGBM模型和SHAP分析：
        
        1. **生成Base64字符串**（在本地Python环境中）：
        ```python
        import base64
        
        # 读取模型文件
        with open('best_model_All_Features_lgb.txt', 'rb') as f:
            model_bytes = f.read()
        
        # 转换为Base64
        model_base64 = base64.b64encode(model_bytes).decode('utf-8')
        print(model_base64)
        ```
        
        2. **将字符串粘贴到app.py的MODEL_BASE64变量中**：
        ```python
        # 模型Base64字符串（请将model_base64.txt的内容粘贴到这里）
        MODEL_BASE64 = "您的base64字符串"
        ```
        
        3. **提交到GitHub并重新部署**
        """)

# 使用说明
with st.expander("📋 使用说明", expanded=False):
    st.info(f"""
    **输入要求：**
    - 临床特征 (2个)：多灶性、T分期
    - 影像组学特征 (3个)：GLCM联合熵、GLRLM灰度非均匀性、形状表面积
    - iTED特征 (2个)：能量、方差
    
    **模型性能：**
    - 内部测试集 AUC-ROC: {MODEL_PERFORMANCE['internal']['AUC-ROC']:.3f} 
      (95% CI: {MODEL_PERFORMANCE['internal']['AUC-ROC_CI'][0]:.3f}-{MODEL_PERFORMANCE['internal']['AUC-ROC_CI'][1]:.3f})
    - 外部验证 AUC-ROC: {MODEL_PERFORMANCE['external']['AUC-ROC']:.3f} 
      (95% CI: {MODEL_PERFORMANCE['external']['AUC-ROC_CI'][0]:.3f}-{MODEL_PERFORMANCE['external']['AUC-ROC_CI'][1]:.3f})
    - 最优阈值: {OPTIMAL_THRESHOLD:.2f} (基于Youden指数)
    - 阈值下敏感性: {MODEL_PERFORMANCE['external']['Sensitivity']*100:.1f}%, 
      特异性: {MODEL_PERFORMANCE['external']['Specificity']*100:.1f}%
    
    **数据特点：**
    - 训练数据：881例患者 (13.5%有远处转移)
    - 使用SMOTE和概率校准处理类别不平衡
    - 外部验证：145例患者 (10.3%有远处转移)
    
    **SHAP分析：**
    - 显示每个特征如何贡献于预测
    - 红色条表示增加风险的特征
    - 蓝色条表示降低风险的特征
    
    **当前状态：**
    - ✅ Python 3.11环境
    - ✅ 所有依赖已安装
    - {'✅ 真实模型已加载' if loaded else '✅ SHAP功能就绪（演示模式）'}
    """)

# 创建输入表单
st.markdown("---")
feature_values = {}

# 使用列布局以获得更好的布局
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 📝 临床特征")

    # 基本信息
    st.markdown("#### 基本信息")
    bcol1, bcol2 = st.columns(2)

    with bcol1:
        multifocal_option = st.selectbox(
            "多灶性 (Multifocal)",
            ["否", "是"],
            help="肿瘤是否为多灶性"
        )
        feature_values['Multifocal'] = "No" if multifocal_option == "否" else "Yes"

    with bcol2:
        feature_values['T_stage'] = st.selectbox(
            "T分期 (T stage)",
            ["T1", "T2", "T3", "T4"],
            index=1,
            help="肿瘤T分期（T1-T4）"
        )

with col2:
    st.markdown("### 🔬 模型特征")
    st.info("""
    该模型结合临床特征、影像组学特征和iTED特征进行综合风险评估。
    此模型在外部验证中达到了最佳性能，AUC-ROC为0.754。
    """)

    # 关键风险因素
    st.markdown("#### 关键风险因素")
    st.markdown("""
    **此模型中的高风险特征：**
    - 多灶性肿瘤
    - 晚期T分期 (T3/T4)
    - 大表面积 (>5000)
    - 高GLCM联合熵 (>6)
    - 高iTED能量值
    - 高iTED方差值
    """)

# 影像组学特征部分
st.markdown("---")
st.markdown("### 📊 影像组学特征")
st.info("影像组学特征是从医学影像中提取的定量测量，捕获肿瘤的纹理和形态特征。")

col1, col2, col3 = st.columns(3)

with col1:
    range_info = FEATURE_RANGES['glcm_JointEntropy']
    feature_values['glcm_JointEntropy'] = st.number_input(
        "GLCM联合熵",
        min_value=range_info['min'],
        max_value=range_info['max'],
        value=range_info['default'],
        step=range_info['step'],
        format=range_info['format'],
        help=range_info['help']
    )

with col2:
    range_info = FEATURE_RANGES['glrlm_GrayLevelNonUniformityNormalized']
    feature_values['glrlm_GrayLevelNonUniformityNormalized'] = st.number_input(
        "GLRLM灰度非均匀性",
        min_value=range_info['min'],
        max_value=range_info['max'],
        value=range_info['default'],
        step=range_info['step'],
        format=range_info['format'],
        help=range_info['help']
    )

with col3:
    range_info = FEATURE_RANGES['shape_SurfaceArea']
    feature_values['shape_SurfaceArea'] = st.number_input(
        "形状表面积",
        min_value=range_info['min'],
        max_value=range_info['max'],
        value=range_info['default'],
        step=range_info['step'],
        format=range_info['format'],
        help=range_info['help']
    )

# iTED特征部分
st.markdown("---")
st.markdown("### 🔬 iTED特征")
st.info("iTED (肿瘤内外差异) 特征反映肿瘤与周围组织之间的差异，是预测转移的重要指标。")

col1, col2 = st.columns(2)

with col1:
    range_info = FEATURE_RANGES['iTED_firstorder_Energy']
    feature_values['iTED_firstorder_Energy'] = st.number_input(
        "iTED能量",
        min_value=range_info['min'],
        max_value=range_info['max'],
        value=range_info['default'],
        step=range_info['step'],
        format=range_info['format'],
        help=range_info['help']
    )

with col2:
    range_info = FEATURE_RANGES['iTED_firstorder_Variance']
    feature_values['iTED_firstorder_Variance'] = st.number_input(
        "iTED方差",
        min_value=range_info['min'],
        max_value=range_info['max'],
        value=range_info['default'],
        step=range_info['step'],
        format=range_info['format'],
        help=range_info['help']
    )

# 预测按钮
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 开始预测", type="primary", use_container_width=True)

if predict_button:
    # 编码分类特征
    encoded_features = encode_categorical_features(feature_values)

    if encoded_features:
        # 创建特征DataFrame
        features_df = pd.DataFrame([encoded_features])[MODEL_CONFIG['features']]

        # 进行预测
        with st.spinner("分析数据中..."):
            probability = predict_risk(model, features_df)

        if probability is not None:
            # 显示结果
            st.markdown("---")
            st.markdown("### 🎯 预测结果")

            # 风险等级确定
            if probability < OPTIMAL_THRESHOLD:
                risk_level = "低风险"
                risk_color = "#28a745"
                recommendation = """
                ✅ **临床建议：**
                - 常规随访即可
                - 每年进行颈部超声和甲状腺球蛋白监测
                - 保持健康生活方式
                - 无需立即进行额外影像检查
                """
            elif probability < 0.3:
                risk_level = "中风险"
                risk_color = "#ffc107"
                recommendation = """
                ⚠️ **临床建议：**
                - 建议加强监测
                - 每6个月随访一次
                - 考虑进行胸部CT基线评估
                - 密切监测甲状腺球蛋白和抗甲状腺球蛋白抗体
                - 可能受益于额外的风险分层
                """
            else:
                risk_level = "高风险"
                risk_color = "#dc3545"
                recommendation = """
                🚨 **临床建议：**
                - 急需进行全面评估
                - 考虑PET-CT扫描检测远处转移
                - 胸部CT和骨扫描指征明确
                - 建议多学科团队会诊
                - 考虑积极的治疗策略
                - 每3-4个月密切监测影像
                """

            # 结果展示
            st.markdown(f"""
            <div class="result-container">
                <h1 style="margin: 0; font-size: 3em;">{probability:.1%}</h1>
                <h2 style="color: white; margin: 10px 0;">远处转移风险概率</h2>
                <h2 style="color: {risk_color}; font-size: 2em;">{risk_level}</h2>
                <p style="color: white; margin-top: 20px;">
                    基于LightGBM模型（7特征）<br>
                    最优阈值: {OPTIMAL_THRESHOLD:.1%} (Youden指数)<br>
                    该阈值下: 敏感性 {MODEL_PERFORMANCE['external']['Sensitivity']*100:.1f}%, 
                    特异性 {MODEL_PERFORMANCE['external']['Specificity']*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

            # 详细指标
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("风险概率", f"{probability:.1%}")
            with col2:
                st.metric("风险等级", risk_level)
            with col3:
                st.metric("模型特征", str(len(MODEL_CONFIG['features'])))

            # 临床建议
            st.markdown("### 💡 临床建议")
            st.markdown(recommendation)

            # SHAP分析
            st.markdown("---")
            display_shap_analysis(model, feature_values, features_df, probability)

            # 风险因素总结
            st.markdown("---")
            st.markdown("### 📋 风险因素总结")

            risk_factors = []
            protective_factors = []

            # 分析风险因素
            if feature_values.get('Multifocal') == 'Yes':
                risk_factors.append("多灶性肿瘤")
            if feature_values.get('T_stage') in ['T3', 'T4']:
                risk_factors.append(f"晚期T分期 ({feature_values['T_stage']})")
            if feature_values.get('glcm_JointEntropy', 0) > 6:
                risk_factors.append(f"高GLCM联合熵 ({feature_values['glcm_JointEntropy']:.2f})")
            if feature_values.get('shape_SurfaceArea', 0) > 5000:
                risk_factors.append(f"大表面积 ({feature_values['shape_SurfaceArea']:.0f})")
            if feature_values.get('iTED_firstorder_Energy', 0) > 1000000:
                risk_factors.append(f"高iTED能量 ({feature_values['iTED_firstorder_Energy']:.0f})")
            if feature_values.get('iTED_firstorder_Variance', 0) > 500:
                risk_factors.append(f"高iTED方差 ({feature_values['iTED_firstorder_Variance']:.1f})")

            # 分析保护因素
            if feature_values.get('Multifocal') == 'No':
                protective_factors.append("单灶性肿瘤")
            if feature_values.get('T_stage') == 'T1':
                protective_factors.append("早期T分期 (T1)")
            if feature_values.get('shape_SurfaceArea', 0) < 1000:
                protective_factors.append(f"小表面积 ({feature_values['shape_SurfaceArea']:.0f})")
            if feature_values.get('glcm_JointEntropy', 0) < 3:
                protective_factors.append(f"低GLCM联合熵 ({feature_values['glcm_JointEntropy']:.2f})")

            col1, col2 = st.columns(2)
            with col1:
                if risk_factors:
                    st.error("**存在的风险因素：**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.info("未发现主要风险因素")

            with col2:
                if protective_factors:
                    st.success("**保护因素：**")
                    for factor in protective_factors:
                        st.write(f"• {factor}")
                else:
                    st.info("未发现保护因素")

            # 生成报告
            st.markdown("### 📄 评估报告")

            report_content = f"""甲状腺癌远处转移风险评估报告
=====================================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
模型: LightGBM多模态融合模型（7特征）

患者信息
---------
多灶性: {'是' if feature_values.get('Multifocal') == 'Yes' else '否'}
T分期: {feature_values.get('T_stage', 'N/A')}

影像组学特征
------------
GLCM联合熵: {feature_values.get('glcm_JointEntropy', 0):.3f}
GLRLM灰度非均匀性: {feature_values.get('glrlm_GrayLevelNonUniformityNormalized', 0):.3f}
形状表面积: {feature_values.get('shape_SurfaceArea', 0):.1f}

iTED特征
--------
能量: {feature_values.get('iTED_firstorder_Energy', 0):.0f}
方差: {feature_values.get('iTED_firstorder_Variance', 0):.1f}

风险评估
--------
远处转移风险概率: {probability:.1%}
风险等级: {risk_level}
使用的模型阈值: {OPTIMAL_THRESHOLD:.1%}

识别的风险因素: {len(risk_factors)}
识别的保护因素: {len(protective_factors)}

临床建议
--------
{recommendation.replace('**', '').replace('✅', '').replace('⚠️', '').replace('🚨', '')}

技术信息
--------
环境: Python {sys.version.split()[0]}
NumPy版本: {np.__version__}
SHAP版本: {shap.__version__}
模式: {'模型已加载' if loaded else '演示模式（SHAP值为模拟）'}
数据来源: 中山医院训练数据 (n=881)
外部验证: 云南大学医院 (n=145)

免责声明
--------
本评估基于机器学习模型，仅供临床参考。
最终诊断和治疗决策应由专业医生基于全面的患者评估做出。

模型性能:
- 内部验证 AUC-ROC: {MODEL_PERFORMANCE['internal']['AUC-ROC']:.3f} 
  (95% CI: {MODEL_PERFORMANCE['internal']['AUC-ROC_CI'][0]:.3f}-{MODEL_PERFORMANCE['internal']['AUC-ROC_CI'][1]:.3f})
- 外部验证 AUC-ROC: {MODEL_PERFORMANCE['external']['AUC-ROC']:.3f} 
  (95% CI: {MODEL_PERFORMANCE['external']['AUC-ROC_CI'][0]:.3f}-{MODEL_PERFORMANCE['external']['AUC-ROC_CI'][1]:.3f})
- 最优阈值: {OPTIMAL_THRESHOLD:.2f} (Youden指数)
- 阈值下敏感性: {MODEL_PERFORMANCE['external']['Sensitivity']*100:.1f}%
- 阈值下特异性: {MODEL_PERFORMANCE['external']['Specificity']*100:.1f}%

医生签名: _____________
日期: _____________
"""

            # 下载按钮
            st.download_button(
                label="📥 下载报告",
                data=report_content,
                file_name=f"thyca_risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# 侧边栏
with st.sidebar:
    st.markdown("### 📊 模型信息")
    st.info(f"""
    **模型状态:** {'已加载 ✅' if loaded else '演示模式 ⚠️'}
    **模型类型:** LightGBM
    **特征总数:** {len(MODEL_CONFIG['features'])}
    - 临床: {MODEL_CONFIG['feature_counts']['clinical']}
    - 影像组学: {MODEL_CONFIG['feature_counts']['radiomics']}
    - iTED: {MODEL_CONFIG['feature_counts']['iTED']}
    
    **性能（外部验证）:**
    - AUC-ROC: {MODEL_PERFORMANCE['external']['AUC-ROC']:.3f}
    - 敏感性: {MODEL_PERFORMANCE['external']['Sensitivity']*100:.1f}%
    - 特异性: {MODEL_PERFORMANCE['external']['Specificity']*100:.1f}%
    - F3-Score: {MODEL_PERFORMANCE['external']['F3-Score']:.3f}
    - MCC: {MODEL_PERFORMANCE['external']['MCC']:.3f}
    
    **模型选择:**
    该模型使用综合评分系统选择，
    权重偏向高敏感性（30%权重）
    以最小化漏诊远处转移病例。
    """)

    st.markdown("### 🎯 SHAP分析")
    if loaded:
        st.success("""
        SHAP (SHapley Additive exPlanations) 
        提供模型可解释性：
        - 显示每个特征的贡献
        - 红色增加风险，蓝色降低风险
        - 帮助理解模型决策过程
        
        **当前状态：**
        - ✅ SHAP包已安装
        - ✅ 模型已加载
        - ✅ 使用真实SHAP分析
        """)
    else:
        st.info("""
        SHAP (SHapley Additive exPlanations) 
        提供模型可解释性：
        - 显示每个特征的贡献
        - 红色增加风险，蓝色降低风险
        - 帮助理解模型决策过程
        
        **当前状态：**
        - ✅ SHAP包已安装
        - ⚠️ 使用演示模式
        - ✅ Plotly可视化工作正常
        
        *提示：在MODEL_BASE64变量中粘贴base64字符串以使用真实模型和准确的SHAP分析*
        """)

    st.markdown("### 📋 特征类别")
    with st.expander("临床特征 (2)"):
        st.markdown("""
        **肿瘤特征：**
        - 多灶性 (Multifocal)
        - T分期 (T_stage)
        """)

    with st.expander("影像组学特征 (3)"):
        st.markdown("""
        **纹理特征：**
        - GLCM联合熵
        - GLRLM灰度非均匀性
        
        **形态特征：**
        - 形状表面积
        """)

    with st.expander("iTED特征 (2)"):
        st.markdown("""
        **一阶统计：**
        - 能量 (Energy)
        - 方差 (Variance)
        
        反映肿瘤内外差异
        """)

    st.markdown("### ⚠️ 使用须知")
    st.warning("""
    1. 仅供医疗专业人员使用
    2. 不能替代临床判断
    3. 基于881例训练数据，M1患病率13.5%
    4. 模型针对高敏感性优化以最小化漏诊病例
    5. 建议定期更新模型
    6. 解释结果时考虑临床背景
    """)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>甲状腺癌远处转移预测系统 v3.2</p>
    <p>基于LightGBM多模态融合模型 with SHAP分析</p>
    <p>模型训练881例患者 | 外部验证145例患者</p>
    <p>© 2025 | 仅供医学研究和临床参考</p>
</div>
""", unsafe_allow_html=True)








