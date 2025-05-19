import streamlit as st
import base64
import pandas as pd
import pickle
import io
import zipfile
import smtplib
from email.message import EmailMessage
import os
from pathlib import Path
import streamlit as st
from tensorflow import keras
from keras import datasets, layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from tensorflow import keras
# from keras import datasets, layers, models Keep these just in case bandaid doesn't work
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from sklearn.preprocessing import LabelEncoder
import joblib

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import requests
import torch.nn.functional as F
from collections import Counter
import cv2
import uuid


# Set page config
st.set_page_config(
    page_title="Nirbhayaa",
    layout="wide",
    page_icon="Nirbhayaa.jpg",
)

models_dir = Path(__file__).parent / "models"

# Add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Home page with feature boxes
def home_page():
    st.markdown("""
        <style>
        .feature-box {
            background-color: rgba(0, 0, 0, 0.5);
            border: 2px solid transparent;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
            height: 100%;
        }
        .feature-box:hover {
            border: 2px solid #00f0ff;
            box-shadow: 0 0 20px #00f0ff;
            transform: scale(1.03);
            cursor: pointer;
        }
        .feature-title {
            font-size: 20px;
            font-weight: bold;
            color: white;
        }
        .feature-desc {
            font-size: 16px;
            color: white;
        }
        .glow {
            font-size: 100px;
            font-weight: bold;
            text-align: center;
            color: white;
            text-shadow: 0 0 9px white, 0 0 20px #00004B, 0 0 20px #00004B;
            margin-bottom: 0px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="glow">Nirbhayaa</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 30px;"><b>Welcome to Nirbhayaa! An Intrusion Detection System for Autonomous Systems.</b></p>', unsafe_allow_html=True)
    st.markdown('<h1 style="text-align: center; font-size: 50px;">What do we Provide?</h1>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container():
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.image("classification.png", width=60)
            st.markdown('<div class="feature-title">DDoS Classification</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-desc">Predict whether the network flow is associated with DDoS or Not.</div>', unsafe_allow_html=True)
            if st.button("Explore Classifier", key="classifier"):
                st.session_state['nav'] = "Romold"
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.image("clustering.png", width=60)
            st.markdown('<div class="feature-title">Anomaly Detection System</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-desc">Analyze Telemetry of Data and Identify Anomalies.</div>', unsafe_allow_html=True)
            if st.button("Detect Anomalies", key="clustering"):
                st.session_state['nav'] = "Tharindu"
            st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        with st.container():
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.image("globular.png", width=60)
            st.markdown('<div class="feature-title">Pasindu Component</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-desc">Detect and study globular clusters in galaxies.</div>', unsafe_allow_html=True)
            if st.button("Explore Globulars", key="globular"):
                st.session_state['nav'] = "Pasindu"
            st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        with st.container():
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.image("enhancer.png", width=60)
            st.markdown('<div class="feature-title">Ransika Component</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-desc">Upscale galaxy images using deep learning GANs.</div>', unsafe_allow_html=True)
            if st.button("Explore Enhancer", key="enhancer"):
                st.session_state['nav'] = "Ransika"
            st.markdown('</div>', unsafe_allow_html=True)

# def ddos_classifier():
#     # Sidebar Navigation (only shown inside this page)
#     page = st.sidebar.radio("📂 Pages", ["Main", "Testing 1", "Testing 2", "Testing 3"])

#     if page == "Main":
#         st.markdown("<h1 style='text-align:center;'>DDoS Detection System 🚀</h1>", unsafe_allow_html=True)
#         st.markdown("<h4 style='text-align:center;'>Provide the input values for the Top 10 SHAP Features below:</h4>", unsafe_allow_html=True)

#         # Load saved model and scaler
#         model = load_model(models_dir / "transformer_top10_model.h5")
#         with open(models_dir / "scaler_ddos.pkl", "rb") as f:
#             scaler = pickle.load(f)
#         with open(models_dir / "top10_features.pkl", "rb") as f:
#             top_10_feature_names = pickle.load(f)

#         # Inputs
#         user_input = {}
#         for feature in top_10_feature_names:
#             user_input[feature] = st.number_input(f"{feature}", value=0.0)

#         # Predict
#         if st.button("🚨 Predict DDoS Attack"):
#             input_values = np.array([list(user_input.values())])
#             input_scaled = scaler.transform(input_values)
#             input_reshaped = np.expand_dims(input_scaled, axis=2)
#             prediction = model.predict(input_reshaped)[0][0]

#             if prediction >= 0.5:
#                 st.error(f"⚠️ DDoS Attack Detected! Probability: {prediction:.2f}")
#             else:
#                 st.success(f"✅ Normal Traffic. Probability: {1 - prediction:.2f}")

#         if st.button("🔙 Back to Home"):
#             st.session_state['nav'] = "Home"

#     elif page == "Testing 1":
#         st.success("🧪 Testing 1 Success")

#     elif page == "Testing 2":
#         st.success("🧪 Testing 2 Success")

#     elif page == "Testing 3":
#         st.success("🧪 Testing 3 Success")

# def ddos_classifier(): ##already working code
#     page = st.sidebar.radio("📂 Pages", [
#         "🔮 Main (Predict)",
#         "📊 SHAP Interpretability",
#         "📈 Model Performance",
#         "📚 DDoS Overview",
#         "⚙️ Feature Engineering"
#     ])

#     if page == "🔮 Main (Predict)":
#         st.markdown("<h1 style='text-align:center;'>DDoS Detection System 🚀</h1>", unsafe_allow_html=True)
#         st.markdown("<h4 style='text-align:center;'>Provide the input values for the Top 10 SHAP Features below:</h4>", unsafe_allow_html=True)

#         # Load model and pre-processing
#         model = load_model(models_dir / "transformer_top10_model.h5")
#         with open(models_dir / "scaler_ddos.pkl", "rb") as f:
#             scaler = pickle.load(f)
#         with open(models_dir / "top10_features.pkl", "rb") as f:
#             top_10_feature_names = pickle.load(f)

#         # Inputs
#         user_input = {feature: st.number_input(f"{feature}", value=0.0) for feature in top_10_feature_names}

#         if st.button("🚨 Predict DDoS Attack"):
#             input_array = np.array([list(user_input.values())])
#             scaled = scaler.transform(input_array)
#             reshaped = np.expand_dims(scaled, axis=2)
#             prediction = model.predict(reshaped)[0][0]

#             if prediction >= 0.5:
#                 st.error(f"⚠️ DDoS Attack Detected! Probability: {prediction:.2f}")
#             else:
#                 st.success(f"✅ Normal Traffic. Probability: {1 - prediction:.2f}")

models_dir = Path("models")  # Update this if your path differs

def ddos_classifier():
    page = st.sidebar.radio("📂 Pages", [
        "🔮 Main (Predict)",
        "📊 SHAP Interpretability",
        "📈 Model Performance",
        "📚 DDoS Overview",
        "⚙️ Feature Engineering"
    ])

    if page == "🔮 Main (Predict)":
        st.markdown("<h1 style='text-align:center;'>DDoS Detection System 🚀</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align:center;'>Upload a CSV file with features used during training:</h4>", unsafe_allow_html=True)

        # Load model and preprocessing tools
        model = load_model(models_dir / "my_transformer_cnn_bilstm_model.h5")
        with open(models_dir / "scaler_ddos1.pkl", "rb") as f:
            scaler = joblib.load(f)

        uploaded_file = st.file_uploader("📄 Upload CSV with input features", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)

            # Optional: Handle any categorical encodings used during training
            for col in ['FlowID', 'SrcIP', 'DstIP']:
                if col in df.columns:
                    df[col] = df[col].astype("category").cat.codes

            # Drop timestamp or other unused columns, if any
            drop_cols = ["timestamp", "Label"]  # Adjust based on what was dropped during training
            df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

            # Convert to NumPy and scale
            try:
                input_array = df.values
                scaled = scaler.transform(input_array)
                reshaped = np.expand_dims(scaled, axis=2)

                if st.button("🚨 Predict DDoS Attack"):
                    predictions = model.predict(reshaped).flatten()
                    for i, prediction in enumerate(predictions):
                        if prediction >= 0.5:
                            st.error(f"⚠️ Row {i+1}: DDoS Attack Detected! Probability: {prediction:.2f}")
                        else:
                            st.success(f"✅ Row {i+1}: Normal Traffic. Probability: {1 - prediction:.2f}")
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")



    elif page == "📊 SHAP Interpretability":
        st.header("📊 SHAP Interpretability")
        st.markdown("See how each feature impacts the prediction using SHAP values.")

        shap_img_path = models_dir / "shap_summary_plot.png"
        if shap_img_path.exists():
            st.image(str(shap_img_path), caption="SHAP Summary Plot", use_column_width=True)
        else:
            st.warning("SHAP visualization not found. Please generate and save `shap_summary_plot.png` in the models directory.")

    elif page == "📈 Model Performance":
        st.header("📈 Model Performance")
        st.markdown("Below is the classification report and evaluation metrics of the model.")

        # Display cached image
        perf_img_path = models_dir / "classification_report.png"
        if perf_img_path.exists():
            st.image(str(perf_img_path), caption="Model Classification Report", use_column_width=True)
        else:
            st.warning("Classification report image not found. Please generate and save `classification_report.png` in the models directory.")

    elif page == "📚 DDoS Overview":
        st.header("📚 What is a DDoS Attack?")
        st.markdown("""
        - **DDoS (Distributed Denial of Service)** attacks flood a system with traffic to make it unavailable.
        - Common types: **UDP Flood**, **SYN Flood**, **HTTP Flood**.
        - Targeted at websites, APIs, networks, servers.
        - Damages: **Downtime**, **data loss**, **revenue loss**, **reputation damage**.
        - This system aims to **detect and mitigate** such attacks using **machine learning**.
        """)
        st.image("https://www.imperva.com/learn/wp-content/uploads/sites/13/2021/02/what-is-ddos-attack.png", use_column_width=True)

    # elif page == "⚙️ Feature Engineering":
    #     st.header("⚙️ Feature Engineering & Model Design")
    #     st.markdown("""
    #     Our system uses the following steps:

    #     - 🔍 **Top 10 features** selected via SHAP.
    #     - 🤖 **Transformer encoder** to transform features based on attention.
    #     - 🧼 Standardized using `scaler_ddos.pkl`.
    #     - 📊 Final output is classified with a dense layer.

    #     **Why Transformers?**
    #     - Captures dependencies and relations between features.
    #     - Good generalization, especially with imbalanced or noisy data.
    #     """)

    #     perf_img_path1 = models_dir / "encoder-layer-norm.png"
    #     st.image(str(perf_img_path1), caption="Transformer Encoder", use_column_width=True)

    elif page == "⚙️ Feature Engineering":
        st.header("⚙️ Feature Engineering & Model Design")

        col1, col2 = st.columns([2, 1])  # Wider text, smaller image

        with col1:
            st.markdown("""
            Our deep learning pipeline includes multiple powerful components designed to extract meaningful patterns from complex network traffic data.

            ### 🔍 Feature Selection with SHAP
            - We use **SHAP (SHapley Additive exPlanations)** to analyze the impact of each feature.
            - The **Top 10 most important features** are selected and saved for focused training.

            ### 🤖 Transformer Encoder Block
            - Introduced early in the network to model **relationships between features**.
            - Key components:
                - **Multi-head Self-Attention**: Learns dependencies between input features.
                - **Feedforward Layers**: Transforms attention outputs.
                - **Residual Connections + Layer Normalization**: Helps learning and stability.

            ### 🔁 Bidirectional LSTM + Attention
            - After the transformer, we stack **Bidirectional LSTMs** to model sequence-level information.
            - An **attention mechanism** highlights the most relevant time steps in the sequence.

            ### 🧼 Standardization & Output
            - Features are standardized using `StandardScaler` and saved (`scaler.pkl`).
            - A final **Dense layer** produces binary classification output (attack vs normal).
            """)

        with col2:
            perf_img_path1 = models_dir / "encoder-layer-norm.png"
            st.image(str(perf_img_path1), caption="Transformer Encoder", use_column_width=True)


## Anomaly Detection Page
def anomaly_detection():

    if st.button("🔙 Back to Home"):
        st.session_state['nav'] = "Home"


    # Email Credentials
    SENDER_EMAIL = "sahasenarathne@gmail.com"
    SENDER_PASSWORD = "izamszaboyrijtgn"  # App password
    DEVELOPER_EMAIL = "tharindusahan11@gmail.com"

    # Load models and thresholds
    with open( models_dir /'angle_scaler.pkl', 'rb') as f:
        angle_scaler = pickle.load(f)

    with open(models_dir/'angle_threshold.pkl', 'rb') as f:
        angle_threshold = pickle.load(f)

    with open(models_dir/'speed_scaler.pkl', 'rb') as f:
        speed_scaler = pickle.load(f)
    with open(models_dir/'speed_threshold.pkl', 'rb') as f:
        speed_threshold = pickle.load(f)

    with open(models_dir/'accl_scaler.pkl', 'rb') as f:
        acceleration_scaler = pickle.load(f)
    with open(models_dir/'accl_threshold.pkl', 'rb') as f:
        acceleration_threshold = pickle.load(f)

    # Anomaly detection functions
    def detect_angle_anomalies(df):
        df = df.drop(['vehicle_id','x','y','speed', 'acceleration', 'density','proximity',
                    'lateral_position','lateral_velocity','lane_boundary_proximity',
                    'time_to_collision','lane'], axis=1)
        df['heading_angle'] = angle_scaler.fit_transform(df[['heading_angle']])
        df['moving_avg'] = df['heading_angle'].rolling(window=2, center=True).mean()
        df['deviation'] = df['heading_angle'] - df['moving_avg']
        df['detected_anomaly'] = df['deviation'].abs() > angle_threshold
        return df

    def detect_speed_anomalies(df):
        df = df.drop(['vehicle_id','x','y','heading_angle', 'acceleration', 'density','proximity',
                    'lateral_position','lateral_velocity','lane_boundary_proximity',
                    'time_to_collision','lane'], axis=1)
        df['speed'] = speed_scaler.fit_transform(df[['speed']])
        df['moving_avg'] = df['speed'].rolling(window=2, center=True).mean()
        df['deviation'] = df['speed'] - df['moving_avg']
        df['detected_anomaly'] = df['deviation'].abs() > speed_threshold
        return df

    def detect_acceleration_anomalies(df):
        df = df.drop(['vehicle_id','x','y','speed', 'heading_angle', 'density','proximity',
                    'lateral_position','lateral_velocity','lane_boundary_proximity',
                    'time_to_collision','lane'], axis=1)
        df['acceleration'] = acceleration_scaler.fit_transform(df[['acceleration']])
        df['moving_avg'] = df['acceleration'].rolling(window=2, center=True).mean()
        df['deviation'] = df['acceleration'] - df['moving_avg']
        df['detected_anomaly'] = df['deviation'].abs() > acceleration_threshold
        return df

    # Composing of Email
    def send_email_with_attachment(to_email, file_data, filename):
        msg = EmailMessage()
        msg['Subject'] = '🚨 Anomaly Report from Streamlit App'
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg.set_content("Hi, your anomaly report is attached.\n\nRegards,\nTeam Nirbhayaa")

        # Add attachment
        msg.add_attachment(file_data, maintype='application', subtype='zip', filename=filename)

        # Send the email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)

    # Streamlit App
    st.title("🚘 Anomaly Detection System")

    st.write("Upload vehicle CSV data to detect anomalies and automatically send the report to the developer.")

    uploaded_file = st.file_uploader("📁 Upload CSV File", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded!")

        with st.spinner("🔍 Detecting anomalies..."):
            angle_results = detect_angle_anomalies(df.copy())
            speed_results = detect_speed_anomalies(df.copy())
            acceleration_results = detect_acceleration_anomalies(df.copy())

        # Create ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for name, data in zip(
                ['angle_anomalies.csv', 'speed_anomalies.csv', 'acceleration_anomalies.csv'],
                [angle_results, speed_results, acceleration_results]
            ):
                csv = io.StringIO()
                data.to_csv(csv, index=False)
                zip_file.writestr(name, csv.getvalue())
        zip_buffer.seek(0)

        # Show previews
        st.subheader("🔍 Preview of Anomaly Detection Results")

        st.write("### Angle Anomalies")
        st.dataframe(angle_results.head())

        st.write("### Speed Anomalies")
        st.dataframe(speed_results.head())

        st.write("### Acceleration Anomalies")
        st.dataframe(acceleration_results.head())

        # Download
        st.download_button("📥 Download ZIP Report", zip_buffer, "anomaly_results.zip", "application/zip")

        # Email to developer
        try:
            send_email_with_attachment(DEVELOPER_EMAIL, zip_buffer.getvalue(), "anomaly_results.zip")
            st.success(f"📧 Report sent to developer at {DEVELOPER_EMAIL}")
        except Exception as e:
            st.error(f"❌ Failed to send email: {e}")


def globular_clusters():
    st.title("Pasindu")
    st.write("✨ This is a placeholder for the globular analysis page.")
    if st.button("🔙 Back to Home"):
        st.session_state['nav'] = "Home"

def image_enhancer():
    st.title("Ransika")
    st.write("🔍 This is a placeholder for the GAN-based enhancer.")
    if st.button("🔙 Back to Home"):
        st.session_state['nav'] = "Home"

# Load background + route based on nav
add_bg_from_local("background.png")
app_mode = st.session_state.get("nav", "Home")

if app_mode == "Home":
    home_page()
elif app_mode == "Romold":
    ddos_classifier()
elif app_mode == "Tharindu":
    anomaly_detection()
elif app_mode == "Pasindu":
    globular_clusters()
elif app_mode == "Ransika":
    image_enhancer()
