import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    layout="wide",
    page_icon="🚗"
)

# -------------------------------
# PREMIUM CUSTOM CSS
# -------------------------------
st.markdown("""
<style>

/* GOOGLE FONT */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* APP BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    min-height: 100vh;
}

/* TITLES */
.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #00f5ff, #ff00ff, #00ff85);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title {
    text-align: center;
    font-size: 1.15rem;
    font-weight: 500;
    color: #d1d5db;
    margin-bottom: 35px;
}

/* CARD */
.card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(14px);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.35);
    margin-bottom: 25px;
}

/* LABELS */
label {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #e5e7eb !important;
}

/* INPUTS */
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 2px solid transparent !important;
    background-image:
        linear-gradient(rgba(255,255,255,0.08), rgba(255,255,255,0.08)),
        linear-gradient(135deg, #00f5ff, #ff00ff, #00ff85);
    background-origin: border-box;
    background-clip: padding-box, border-box;
    transition: all 0.3s ease-in-out;
}

/* INPUT FOCUS */
.stNumberInput input:focus,
.stSelectbox div[data-baseweb="select"]:focus-within {
    box-shadow: 0 0 18px rgba(0,245,255,0.8);
    transform: scale(1.02);
}

/* ACCURACY BOX */
.accuracy-box {
    font-size: 28px;
    font-weight: 700;
    text-align: center;
    padding: 25px;
    border-radius: 16px;
    color: white;
    background: linear-gradient(135deg, #11998e, #38ef7d);
    box-shadow: 0 0 22px rgba(56,239,125,0.7);
}

/* BUTTON */
.stButton > button {
    width: 100%;
    padding: 14px;
    font-size: 1.1rem;
    font-weight: 700;
    border-radius: 14px;
    border: none;
    color: white;
    background: linear-gradient(135deg, #ff512f, #dd2476);
    box-shadow: 0 8px 25px rgba(221,36,118,0.6);
    transition: all 0.25s ease-in-out;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 30px rgba(255,81,47,0.9);
}

/* PRICE BOX */
.price-box {
    font-size: 2.5rem;
    font-weight: 800;
    text-align: center;
    padding: 35px;
    border-radius: 22px;
    color: white;
    background: linear-gradient(135deg, #667eea, #764ba2);
    box-shadow: 0 0 40px rgba(118,75,162,0.9);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141e30, #243b55);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.markdown('<div class="main-title">🚗 Car Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Compare ML Models & Predict Selling Price</div>', unsafe_allow_html=True)

# -------------------------------
# MODEL ACCURACY
# -------------------------------
model_accuracy = {
    "Linear Regression": 88.0,
    "Ridge Regression": 89.3,
    "Lasso Regression": 87.1,
    "Decision Tree": 91.0,
    "Random Forest": 95.2,
    "Gradient Boosting": 94.1
}

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("🔧 Model Selection")

model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(model_accuracy.keys())
)

model_file_map = {
    "Linear Regression": "saved_models/Linear_Regression.pkl",
    "Ridge Regression": "saved_models/Ridge_Regression.pkl",
    "Lasso Regression": "saved_models/Lasso_Regression.pkl",
    "Decision Tree": "saved_models/Decision_Tree.pkl",
    "Random Forest": "saved_models/Random_Forest.pkl",
    "Gradient Boosting": "saved_models/Gradient_Boosting.pkl"
}

model = pickle.load(open(model_file_map[model_name], "rb"))

# -------------------------------
# MODEL INFO
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f'<div class="card"><h3>📌 Selected Model</h3><h2>{model_name}</h2></div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f'<div class="accuracy-box">🎯 Accuracy<br>{model_accuracy[model_name]}%</div>',
        unsafe_allow_html=True
    )

# -------------------------------
# INPUT SECTION
# -------------------------------
st.markdown("## 🔢 Enter Car Details")

st.markdown('<div class="card">', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)
with col3:
    year = st.number_input("Year", 2000, 2025, 2015)
with col4:
    present_price = st.number_input("Present Price (Lakhs)", 0.0, 50.0, 5.0)
with col5:
    kms_driven = st.number_input("Kilometers Driven", 0, 300000, 30000)

col6, col7, col8, col9 = st.columns(4)
with col6:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
with col7:
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
with col8:
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
with col9:
    owner = st.selectbox("Owner", [0, 1, 3])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict Price"):
    input_df = pd.DataFrame(
        [[year, present_price, kms_driven,
          fuel_type, seller_type, transmission, owner]],
        columns=[
            "Year", "Present_Price", "Kms_Driven",
            "Fuel_Type", "Seller_Type", "Transmission", "Owner"
        ]
    )

    prediction = model.predict(input_df)[0]

    st.markdown("## 💰 Predicted Selling Price")
    st.markdown(
        f'<div class="price-box">₹ {prediction:.2f} Lakhs</div>',
        unsafe_allow_html=True
    )
