import streamlit as st
import pandas as pd
import sys
import os

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import CreditRiskPredictor

st.set_page_config(page_title="Credit Risk Predictor", page_icon="🏦", layout="wide")

# Custom CSS for banking style
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        color: white;
        border-radius: 8px;
        border: none;
        width: 100%;
        font-weight: 800;
        font-size: 18px;
        padding: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255, 65, 108, 0.6);
        background: linear-gradient(90deg, #ff4b2b, #ff416c);
        color: white;
    }
    
    .metric-card {
        background: rgba(30, 30, 46, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        color: #ffffff;
        padding: 30px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    .risk-low {
        background: -webkit-linear-gradient(#00b09b, #96c93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 48px;
        margin: 10px 0;
        display: inline-block;
        animation: pulse 2s infinite ease-in-out;
    }
    .risk-medium {
        background: -webkit-linear-gradient(#f7b733, #fc4a1a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 48px;
        margin: 10px 0;
        display: inline-block;
    }
    .risk-high {
        background: -webkit-linear-gradient(#ff416c, #ff4b2b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 48px;
        margin: 10px 0;
        display: inline-block;
        animation: pulse-danger 1.5s infinite ease-in-out;
    }
    
    .prob-text {
        background: -webkit-linear-gradient(#00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 48px;
        margin: 10px 0;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }
    
    @keyframes pulse-danger {
        0% { transform: scale(1); filter: drop-shadow(0 0 5px rgba(255,65,108,0.2)); }
        50% { transform: scale(1.06); filter: drop-shadow(0 0 15px rgba(255,65,108,0.6)); }
        100% { transform: scale(1); filter: drop-shadow(0 0 5px rgba(255,65,108,0.2)); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return CreditRiskPredictor()

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure you have run the training pipeline.")
    st.stop()

st.title("🏦 AI-Powered Credit Risk Prediction System")
st.markdown("Predict whether a customer is likely to default on a loan based on financial and personal attributes.")

# Sidebar for inputs
st.sidebar.header("Customer Profile")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
job = st.sidebar.selectbox("Job Skill Level", [0, 1, 2, 3], format_func=lambda x: {0: "Unskilled / Non-resident", 1: "Unskilled / Resident", 2: "Skilled", 3: "Highly Skilled"}[x])
housing = st.sidebar.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.sidebar.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich", "no_account"])
checking_account = st.sidebar.selectbox("Checking Account", ["little", "moderate", "rich", "no_account"])

st.sidebar.header("Loan Details")
credit_amount = st.sidebar.number_input("Credit Amount (DM)", min_value=100, max_value=50000, value=2500)
duration = st.sidebar.slider("Duration (Months)", 4, 72, 24)
purpose = st.sidebar.selectbox("Purpose", ["car", "furniture/equipment", "radio/TV", "domestic appliances", "repairs", "education", "business", "vacation/others"])

if st.sidebar.button("Predict Risk"):
    input_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_account,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose
    }])
    
    with st.spinner("Analyzing credit risk..."):
        prob, category, X_processed = predictor.predict(input_data)
        top_risk, top_safe = predictor.explain_prediction(X_processed)
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: #a0a0b0; font-weight: 400; margin-bottom: 5px;">Default Probability</h3>
            <div class="prob-text">{prob * 100:.2f}%</div>
        </div>
        ''', unsafe_allow_html=True)
        
    with col2:
        if category == "Low Risk":
            cat_html = f'<span class="risk-low">{category}</span>'
        elif category == "Medium Risk":
            cat_html = f'<span class="risk-medium">{category}</span>'
        else:
            cat_html = f'<span class="risk-high">{category}</span>'
            
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: #a0a0b0; font-weight: 400; margin-bottom: 5px;">Risk Category</h3>
            {cat_html}
        </div>
        ''', unsafe_allow_html=True)
        
    st.markdown("### Model Explainability (Feature Impacts)")
    st.info("The model determines risk based on how these features deviate from the baseline. Here are the top factors for this prediction:")
    
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.success("✅ **Top Factors Decreasing Risk**")
        for factor in top_safe:
            st.write(f"- {factor}")
            
    with col_exp2:
        st.error("⚠️ **Top Factors Increasing Risk**")
        for factor in top_risk:
            st.write(f"- {factor}")

    st.markdown("#### Local Feature Impacts Plot")
    try:
        st.image(r"d:\ds project\credit-risk-project\visuals\shap_local_explanation.png", use_container_width=True)
    except Exception as e:
        st.warning("Local explanation plot not found.")
