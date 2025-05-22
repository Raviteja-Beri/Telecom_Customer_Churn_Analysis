import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Set page configuration for a professional look
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

# Load the trained model
try:
    model = joblib.load('final_gb_classifier.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'final_gb_classifier.pkl' is in the correct directory.")
    st.stop()

# Function to preprocess input data
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({
        'Electronic check': 0, 
        'Mailed check': 1, 
        'Bank transfer (automatic)': 2, 
        'Credit card (automatic)': 3
    })
    return df

# Custom CSS with fixes for radio button labels, number input headings, and selectbox headings
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stNumberInput, .stSelectbox, .stRadio {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    /* Fix for radio button label visibility */
    div.stRadio > div > label > div > p {
        color: #000000 !important; /* Black color for maximum contrast */
        font-weight: 500;
    }
    div.stRadio > div {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    div.stRadio label p {
        color: #000000 !important;
    }
    /* Fix for number input headings (labels) visibility */
    div.stNumberInput > label {
        color: #000000 !important; /* Black color for number input labels */
        font-weight: 600;
        font-size: 16px;
    }
    /* Fix for selectbox headings (labels) visibility */
    div.stSelectbox > label {
        color: #000000 !important; /* Black color for selectbox labels */
        font-weight: 600;
        font-size: 16px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .churn {
        background-color: #ffcccc;
        color: #cc0000;
    }
    .stay {
        background-color: #ccffcc;
        color: #006600;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Predict whether a customer will stay or churn with our advanced machine learning model! Input the details below to get started.")

# Organize inputs in columns for a cleaner layout
st.subheader("Customer Information")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.radio("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    senior_citizen = st.radio("Senior Citizen", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    partner = st.radio("Partner", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    dependents = st.radio("Dependents", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col2:
    phone_service = st.radio("Phone Service", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    multiple_lines = st.radio("Multiple Lines", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    internet_service = st.selectbox("Internet Service", options=['DSL', 'Fiber optic', 'No'])
    contract = st.selectbox("Contract Type", options=['Month-to-month', 'One year', 'Two year'])

with col3:
    online_security = st.radio("Online Security", options=[0, 1, 2], format_func=lambda x: {0: "No", 1: "Yes", 2: "No Internet"}[x])
    online_backup = st.radio("Online Backup", options=[0, 1, 2], format_func=lambda x: {0: "No", 1: "Yes", 2: "No Internet"}[x])
    device_protection = st.radio("Device Protection", options=[0, 1, 2], format_func=lambda x: {0: "No", 1: "Yes", 2: "No Internet"}[x])
    tech_support = st.radio("Tech Support", options=[0, 1, 2], format_func=lambda x: {0: "No", 1: "Yes", 2: "No Internet"}[x])

# Additional inputs
st.subheader("Billing and Subscription Details")
col4, col5 = st.columns(2)

with col4:
    streaming_tv = st.radio("Streaming TV", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    streaming_movies = st.radio("Streaming Movies", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    paperless_billing = st.radio("Paperless Billing", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col5:
    payment_method = st.selectbox("Payment Method", options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=0.1)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0, step=1.0)
    tenure_group = st.number_input("Tenure (Months)", min_value=0, value=12, step=1)

# Make prediction
if st.button("🔍 Predict Churn"):
    # Create dictionary from user inputs
    user_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_group': tenure_group
    }
    
    # Preprocess input data
    processed_data = preprocess_input(user_data)
    
    # Make prediction and get probability
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[0]
    
    # Display prediction result
    if prediction[0] == 1:
        st.markdown('<div class="prediction-box churn">⚠️ The customer is likely to churn!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-box stay">✅ The customer is likely to stay!</div>', unsafe_allow_html=True)
    
    # Display prediction probability with a bar chart
    st.subheader("Prediction Confidence")
    prob_df = pd.DataFrame({
        'Outcome': ['Stay', 'Churn'],
        'Probability': [probability[0], probability[1]]
    })
    fig = px.bar(prob_df, x='Outcome', y='Probability', color='Outcome', 
                 color_discrete_map={'Stay': '#006600', 'Churn': '#cc0000'},
                 title="Prediction Probability", height=400)
    fig.update_layout(showlegend=False, yaxis_title="Probability", xaxis_title="Prediction")
    st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit | Powered by xAI's Machine Learning")