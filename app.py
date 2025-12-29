import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    with open('breast_cancer_clinical_model.pkl', 'rb') as f:
        model_package = pickle.load(f)
    return model_package

model_package = load_model()

# Page config
st.set_page_config(
    page_title="Breast Cancer Risk Predictor",
    page_icon="🎗️",
    layout="wide"
)

# Title
st.title("🎗️ Breast Cancer 10-Year Mortality Risk Prediction")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("Patient Information")

# Collect inputs
age = st.sidebar.number_input("Age at Diagnosis", min_value=20, max_value=100, value=55)
tumor_size = st.sidebar.number_input("Tumor Size (mm)", min_value=1, max_value=200, value=20)
tumor_stage = st.sidebar.selectbox("Tumor Stage", [0, 1, 2, 3, 4])
lymph_nodes = st.sidebar.number_input("Lymph Nodes Examined Positive", min_value=0, max_value=50, value=0)
grade = st.sidebar.selectbox("Histologic Grade", [1, 2, 3])

er_status = st.sidebar.selectbox("ER Status", ["Positive", "Negative"])
pr_status = st.sidebar.selectbox("PR Status", ["Positive", "Negative"])
her2_status = st.sidebar.selectbox("HER2 Status", ["Positive", "Negative"])

chemo = st.sidebar.selectbox("Chemotherapy", ["Yes", "No"])
hormone = st.sidebar.selectbox("Hormone Therapy", ["Yes", "No"])
radio = st.sidebar.selectbox("Radio Therapy", ["Yes", "No"])

# Predict button
# Replace the "Prepare input data" section in app.py with this:

if st.sidebar.button("🔮 Predict Risk", type="primary"):
    
    # Prepare input data IN THE SAME ORDER as training
    input_data = pd.DataFrame({
        'Age at Diagnosis': [age],
        'Tumor Size': [tumor_size],
        'Tumor Stage': [tumor_stage],
        'Lymph nodes examined positive': [lymph_nodes],
        'ER Status': [er_status],
        'PR Status': [pr_status],
        'HER2 Status': [her2_status],
        'Neoplasm Histologic Grade': [grade],
        'Chemotherapy': [chemo],
        'Hormone Therapy': [hormone],
        'Radio Therapy': [radio]
    })
    
    # Reorder columns to match training order
    input_data = input_data[model_package['feature_names']]
    
    # Rest of the code stays the same...
    
    # Preprocess
    # Impute numerical
    input_data[model_package['numerical_features']] = model_package['num_imputer'].transform(
        input_data[model_package['numerical_features']]
    )
    
    # Impute categorical
    input_data[model_package['categorical_features']] = model_package['cat_imputer'].transform(
        input_data[model_package['categorical_features']]
    )
    
    # Encode categorical
    for col in model_package['categorical_features']:
        le = model_package['label_encoders'][col]
        val = input_data[col].values[0]
        if val in le.classes_:
            input_data[col] = le.transform([val])
        else:
            input_data[col] = le.transform([le.classes_[0]])
    
    # Scale
    input_scaled = model_package['scaler'].transform(input_data)
    
    # Predict
    probability = model_package['model'].predict_proba(input_scaled)[0, 1]
    prediction = int(probability >= model_package['threshold'])
    
    # Display results
    st.markdown("---")
    st.header("📊 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Risk Probability", f"{probability*100:.1f}%")
    
    with col2:
        risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
        color = "🔴" if prediction == 1 else "🟢"
        st.metric("Risk Level", f"{color} {risk_level}")
    
    with col3:
        survival_prob = (1 - probability) * 100
        st.metric("10-Year Survival Probability", f"{survival_prob:.1f}%")
    
    # Risk interpretation
    st.markdown("---")
    st.subheader("📋 Clinical Interpretation")
    
    if prediction == 1:
        st.error("⚠️ **High Risk Patient**: This patient has elevated risk of mortality within 10 years. Consider intensive monitoring and aggressive treatment options.")
    else:
        st.success("✅ **Low Risk Patient**: This patient has favorable prognosis. Standard treatment protocols recommended with regular follow-ups.")
    
    # Display input summary
    st.markdown("---")
    st.subheader("📝 Patient Profile Summary")
    st.dataframe(input_data.T, use_container_width=True)

else:
    st.info("👈 Please enter patient information in the sidebar and click 'Predict Risk'")
    
    # Show example
    st.markdown("---")
    st.subheader("ℹ️ About This Tool")
    st.write("""
    This tool predicts the 10-year mortality risk for breast cancer patients using machine learning.
    
    **Features Used:**
    - Patient demographics (Age)
    - Tumor characteristics (Size, Stage, Grade)
    - Biomarkers (ER, PR, HER2 status)
    - Treatment information (Chemotherapy, Hormone, Radiation)
    - Lymph node involvement
    
    **Model Performance:**
    - Trained on METABRIC dataset
    - Uses optimized threshold (0.3) for better sensitivity
    - Achieves high accuracy in predicting 10-year outcomes
    """)