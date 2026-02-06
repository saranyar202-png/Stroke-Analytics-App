
import streamlit as st

st.set_page_config(
    page_title="Stroke Analytics App",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
/* Background */
.main {
    background-color: #f9fbfd;
}

/* Title */
h1 {
    color: #2c3e50;
    font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1f2937;
}
section[data-testid="stSidebar"] * {
    color: white;
}

/* Buttons */
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-size: 16px;
}
div.stButton > button:hover {
    background-color: #1d4ed8;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import joblib
import os
import joblib

if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")
else:
    st.error("Model file not found")
    st.stop()
    
if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")
else:
    st.error("Model file not found")
    st.stop()
import joblib

model = joblib.load("model.pkl")

import pickle

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Stroke Analytics App", layout="wide")

st.title("🧠 Stroke Prediction & Data Analytics")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🧠 Stroke Analytics")
st.sidebar.markdown("Healthcare Data • ML • Analytics")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Analytics", "📁 Dataset", "🤖 Prediction", "🧠 Insights"]
)

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.markdown("## 🧠 Stroke Risk Analytics Platform")
    st.write(
        "An interactive healthcare analytics web application that visualizes "
        "stroke risk factors and predicts stroke probability using machine learning."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", df.shape[0])
    col2.metric("Stroke Cases", int(df['stroke'].sum()))
    col3.metric("Stroke Rate (%)", round(df['stroke'].mean()*100, 2))

    st.markdown("### 🔍 What this app does")
    st.write("""
    • Analyzes healthcare data  
    • Visualizes stroke risk factors  
    • Predicts stroke probability  
    • Supports data-driven medical decisions  
    """)

# ---------------- DATASET ----------------
elif page == "📁 Dataset":
    st.markdown("## 📁 Healthcare Stroke Dataset")
    st.write(
        "This dataset contains patient health information used to analyze "
        "stroke risk factors and build a prediction model."
    )

    st.markdown("### 🔍 Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### 📊 Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Stroke Cases", int(df["stroke"].sum()))

    st.markdown("### 🧾 Column Information")
    st.dataframe(
        df.dtypes.rename("Data Type").reset_index().rename(columns={"index": "Column"}),
        use_container_width=True
    )

    st.markdown("### ⬇️ Download Dataset")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name="stroke_dataset.csv",
        mime="text/csv"
    )

# ---------------- ANALYSIS ----------------
elif page == "📊 Analytics":
    st.markdown("## 📊 Data Analytics Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stroke Distribution")
        st.bar_chart(df["stroke"].value_counts())

    with col2:
        st.subheader("Hypertension Impact")
        st.bar_chart(df.groupby("hypertension")["stroke"].mean())

    st.subheader("Age vs Stroke Probability")
    st.line_chart(df.groupby("age")["stroke"].mean())

# ---------------- PREDICTION ----------------
elif page == "🤖 Prediction":
    st.markdown("## 🤖 Stroke Risk Prediction")
    st.info("Enter patient details to assess stroke risk")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 1, 100, 45)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])

    with col2:
        avg_glucose = st.number_input("Average Glucose Level", 50.0, 300.0)
        bmi = st.number_input("BMI", 10.0, 60.0)

    if st.button("Predict Stroke Risk"):
        prediction = model.predict([[age, hypertension, heart_disease, avg_glucose, bmi]])
        if prediction[0] == 1:
            st.error("⚠️ High Risk of Stroke")
        else:
            st.success("✅ Low Risk of Stroke")

# ---------------- INSIGHTS ----------------
elif page == "🧠 Insights":
    st.markdown("## 🧠 Key Insights & Findings")

    st.success("""
    ✔ Stroke risk increases significantly with age  
    ✔ Hypertension and high glucose levels are major risk factors  
    ✔ Data analytics enables early identification of high-risk patients  
    """)

    st.markdown("### 📌 Business / Healthcare Impact")
    st.write("""
    • Supports preventive healthcare  
    • Assists doctors in decision-making  
    • Improves patient risk monitoring  

    """)


