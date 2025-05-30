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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
            #st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.image("ddos.png", width=60)
            st.markdown('<div class="feature-title">DDoS Classification</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-desc">Predict whether the network flow is associated with DDoS or Not.</div>', unsafe_allow_html=True)
            if st.button("Explore DDOS Classifier", key="classifier"):
                st.session_state['nav'] = "Romold"
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.container():
            #st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.image("anomaly.png", width=60)
            st.markdown('<div class="feature-title">Anomaly Detection System</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-desc">Analyze Telemetry of Data, Identify Anomalies and Send Anomaly Report.</div>', unsafe_allow_html=True)
            if st.button("Explore Anomlay Detector", key="clustering"):
                st.session_state['nav'] = "Tharindu"
            st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        with st.container():
            #st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.image("lidar.png", width=60)
            st.markdown('<div class="feature-title">LiDAR Spoofing Component</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-desc">Identify LiDAR Sensor Spoofing Attempts In The Autonomous Vehicle.</div>', unsafe_allow_html=True)
            if st.button("Explore Detector", key="globular"):
                st.session_state['nav'] = "Pasindu"
            st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        with st.container():
            #st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.image("camera.png", width=60)
            st.markdown('<div class="feature-title">Camera Spoofing Component</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-desc">Identify Camera Spoofing Adversarial Attack Attempts In The Autonomous Vehicles.</div>', unsafe_allow_html=True)
            if st.button("Explore Detector", key="enhancer"):
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
        "📚 DDoS Overview",
        "📈 Model Performance",
        "⚙️ Feature Engineering",
        "🔮 Main (Predict)"
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

        if st.button("🔙 Back to Home"):
            st.session_state['nav'] = "Home"



    # elif page == "📊 SHAP Interpretability":
    #     st.header("📊 SHAP Interpretability")
    #     st.markdown("See how each feature impacts the prediction using SHAP values.")

    #     shap_img_path = models_dir / "shap_summary_plot.png"

    #     if shap_img_path.exists():
    #         st.image(str(shap_img_path), caption="SHAP Summary Plot", use_column_width=True)
    #     else:
    #         st.warning("SHAP visualization not found. Please generate and save `shap_summary_plot.png` in the models directory.")
    
    #     if st.button("🔙 Back to Home"):
    #         st.session_state['nav'] = "Home"

    elif page == "📈 Model Performance":
        st.header("📈 Model Performance")
        st.markdown("Below is the classification report and evaluation metrics of the model.")

        # Display cached image
        perf_img_path = models_dir / "classification_ddos.png"
        if perf_img_path.exists():
            st.image(str(perf_img_path), caption="Model Classification Report", use_column_width=True)
        else:
            st.warning("Classification report image not found. Please generate and save `classification_report.png` in the models directory.")

        if st.button("🔙 Back to Home"):
                st.session_state['nav'] = "Home"

    # elif page == "📚 DDoS Overview":
    #     st.header("📚 What is a DDoS Attack?")
    #     st.markdown("""
    #     - **DDoS (Distributed Denial of Service)** attacks flood a system with traffic to make it unavailable.
    #     - Common types: **UDP Flood**, **SYN Flood**, **HTTP Flood**.
    #     - Targeted at websites, APIs, networks, servers.
    #     - Damages: **Downtime**, **data loss**, **revenue loss**, **reputation damage**.
    #     - This system aims to **detect and mitigate** such attacks using **machine learning**.
    #     """)

    #     if st.button("🔙 Back to Home"):
    #         st.session_state['nav'] = "Home"
    elif page == "📚 DDoS Overview":
        st.header("🚨 Understanding DDoS Attacks")

        st.markdown("""
        ## 🌐 What is a DDoS Attack?

        A **Distributed Denial of Service (DDoS)** attack occurs when multiple systems overwhelm a target (like a website or server) with a flood of traffic, causing it to slow down or crash completely.

        ---

        ### 💣 Common Types of DDoS Attacks:
        - 🔸 **UDP Flood**: Overloads systems with User Datagram Protocol packets.
        - 🔸 **SYN Flood**: Exploits the handshake process in TCP connections.
        - 🔸 **HTTP Flood**: Mimics legitimate web traffic to exhaust server resources.

        ---

        ### 🎯 Typical Targets:
        - 🖥️ Websites and web applications  
        - 🌐 APIs and backend systems  
        - 🧠 Networks and DNS servers  

        ---

        ### 🧨 Consequences of a DDoS Attack:
        - ❌ **Downtime** – service becomes unavailable to users  
        - 📉 **Revenue loss** – especially during peak traffic  
        - 🔓 **Data loss** – if coupled with a breach  
        - 💔 **Reputation damage** – loss of user trust  

        """)

        if st.button("🔙 Back to Home"):
            st.session_state['nav'] = "Home"


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

        if st.button("🔙 Back to Home"):
            st.session_state['nav'] = "Home"

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

    
    def plot_confusion_matrix(df, predicted_col, title, width=5, height=4):
        y_true = df['label']  # 0 = normal, 1 = anomaly
        y_pred = df[predicted_col].map({False: 0, True: 1})  # Convert to 0/1
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(width, height))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
        disp.plot(ax=ax, values_format='d')
        ax.set_title(title)
        #st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.image(buf, width=500) 
    
    
    
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

        if 'label' in angle_results.columns:
            plot_confusion_matrix(angle_results, 'detected_anomaly', "Confusion Matrix - Angle")

        st.write("### Speed Anomalies")
        st.dataframe(speed_results.head())

        if 'label' in speed_results.columns:
            plot_confusion_matrix(speed_results, 'detected_anomaly', "Confusion Matrix - Speed")


        st.write("### Acceleration Anomalies")
        st.dataframe(acceleration_results.head())

        if 'label' in acceleration_results.columns:
            plot_confusion_matrix(acceleration_results, 'detected_anomaly', "Confusion Matrix - Acceleration")

        # Download
        st.download_button("📥 Download ZIP Report", zip_buffer, "anomaly_results.zip", "application/zip")

        # Email to developer
        try:
            send_email_with_attachment(DEVELOPER_EMAIL, zip_buffer.getvalue(), "anomaly_results.zip")
            st.success(f"📧 Report sent to developer at {DEVELOPER_EMAIL}")
        except Exception as e:
            st.error(f"❌ Failed to send email: {e}")


def Lidar_Spoofing_Detection():
    st.title("🚗 LiDAR Spoofing Detection")

    UPLOAD_FOLDER = "uploads"
    RESULT_FOLDER = "static/results"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    model1 = load_model(str(models_dir / "classification_autoencoder_test1.keras"))
    model2 = load_model(str(models_dir / "jammed_classification_autoencoder_test1.keras"))
    


    def load_point_cloud(bin_path):
        point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return point_cloud[:, :3]

    def point_cloud_to_height_bev(point_cloud, res=0.1, x_range=(-50, 50), y_range=(-50, 50), z_range=(-2, 2)):
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range

        mask = (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max) & \
               (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] <= y_max) & \
               (point_cloud[:, 2] >= z_min) & (point_cloud[:, 2] <= z_max)
        filtered_pc = point_cloud[mask]

        x_img = ((filtered_pc[:, 0] - x_min) / res).astype(np.int32)
        y_img = ((filtered_pc[:, 1] - y_min) / res).astype(np.int32)

        bev_size_x = int((x_max - x_min) / res)
        bev_size_y = int((y_max - y_min) / res)

        bev_img = np.zeros((bev_size_y, bev_size_x))
        norm_heights = (filtered_pc[:, 2] - z_min) / (z_max - z_min)

        x_img = np.clip(x_img, 0, bev_size_x - 1)
        y_img = np.clip(y_img, 0, bev_size_y - 1)

        bev_img[y_img, x_img] = norm_heights

        return bev_img

    def preprocess_image(image_path, target_size=(200, 200)):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def crop_center(img, crop_x, crop_y):
        h, w = img.shape[:2]
        start_x = w // 2 - (crop_x // 2)
        start_y = h // 2 - (crop_y // 2)
        return img[start_y:start_y+crop_y, start_x:start_x+crop_x]

    def process_file(file, model, crop=False, prefix=''):
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        point_cloud = load_point_cloud(file_path)
        bev_image = point_cloud_to_height_bev(point_cloud)

        original_image_path = os.path.join(RESULT_FOLDER, f"{prefix}_original.png")
        plt.imsave(original_image_path, bev_image, cmap="jet", origin="lower")

        if crop:
            img = cv2.imread(original_image_path)
            cropped_img = crop_center(img, crop_x=450, crop_y=160)
            bev_image_path = os.path.join(RESULT_FOLDER, f"{prefix}_cropped.png")
            cv2.imwrite(bev_image_path, cropped_img)
        else:
            bev_image_path = original_image_path

        input_image = preprocess_image(bev_image_path)
        reconstruction, classification_output = model.predict(input_image)
        predicted_class = np.argmax(classification_output[0])
        label = "Spoofed" if predicted_class == 1 else "Genuine"

        return bev_image_path, label

    tab1, tab2 = st.tabs(["Model 1", "Model 2"])

    with tab1:
        st.subheader("Upload for Model 1 (No Cropping)")
        file1 = st.file_uploader("Choose File 1 (.bin)", type=['bin'], key='file1')
        file2 = st.file_uploader("Choose File 2 (.bin)", type=['bin'], key='file2')

        if file1:
            img_path, label = process_file(file1, model1, crop=False, prefix='file1')
            st.image(img_path, caption=f"File 1 - Prediction: {label}", width=300)
            st.success(f"Prediction: {label}")

        if file2:
            img_path, label = process_file(file2, model1, crop=False, prefix='file2')
            st.image(img_path, caption=f"File 2 - Prediction: {label}", width=300)
            st.success(f"Prediction: {label}")

    with tab2:
        st.subheader("Upload for Model 2 (With Cropping)")
        file3 = st.file_uploader("Choose File 3 (.bin)", type=['bin'], key='file3')
        file4 = st.file_uploader("Choose File 4 (.bin)", type=['bin'], key='file4')

        if file3:
            img_path, label = process_file(file3, model2, crop=False, prefix='file3')
            st.image(img_path, caption=f"File 3 - Prediction: {label}", width=300)
            st.success(f"Prediction: {label}")

        if file4:
            img_path, label = process_file(file4, model2, crop=False, prefix='file4')
            st.image(img_path, caption=f"File 4 - Prediction: {label}", width=300)
            st.success(f"Prediction: {label}")
    
    if st.button("🔙 Back to Home"):
        st.session_state['nav'] = "Home"




### camera spoofing adversarial attacks detection component------------------------------------------------------------------------------

# Cache ImageNet labels
@st.cache_data
def download_imagenet_classes():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    return [line.strip() for line in response.text.splitlines()]

# Cache GoogleNet model
@st.cache_resource
def load_googlenet():
    model = models.googlenet(pretrained=True)
    model.eval()
    return model

# Initialize global variables
labels = download_imagenet_classes()
googlenet = load_googlenet()
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Denoising Algorithms
def fast_non_local_means_denoising_algorithm(image):
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    denoised_image = Image.fromarray(denoised_image)
    denoised_tensor = preprocess(denoised_image).unsqueeze(0)
    with torch.no_grad():
        output = googlenet(denoised_tensor)
        predicted_idx = output.argmax().item()
    return labels[predicted_idx]

def midpoint_filter(image, kernel_size=3):
    H, W, C = image.shape
    pad = kernel_size // 2
    filtered_image = np.zeros_like(image, dtype=np.float32)
    for ch in range(C):
        padded_channel = cv2.copyMakeBorder(image[:, :, ch], pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        for i in range(H):
            for j in range(W):
                region = padded_channel[i:i + kernel_size, j:j + kernel_size].flatten()
                min_val = np.min(region)
                max_val = np.max(region)
                midpoint_value = (min_val + max_val) / 2
                filtered_image[i, j, ch] = midpoint_value
    filtered_image = filtered_image.astype(np.uint8)
    clean_image = Image.fromarray(filtered_image)
    clean_tensor = preprocess(clean_image).unsqueeze(0)
    with torch.no_grad():
        output = googlenet(clean_tensor)
        predicted_idx = output.argmax().item()
    return labels[predicted_idx]

def bilateral_filtering_algorithm(image):
    clean_image = cv2.bilateralFilter(image, d=15, sigmaColor=75, sigmaSpace=75)
    clean_image = Image.fromarray(clean_image)
    clean_tensor = preprocess(clean_image).unsqueeze(0)
    with torch.no_grad():
        output = googlenet(clean_tensor)
        predicted_idx = output.argmax().item()
    return labels[predicted_idx]

def geometric_mean_filter(img, kernel_size=3):
    pad = kernel_size // 2
    padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    filtered_img = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded_img[i:i+kernel_size, j:j+kernel_size].flatten()
            product = np.prod(window)
            geom_mean = product ** (1 / len(window))
            filtered_img[i, j] = geom_mean
    return filtered_img.astype(np.uint8)

def geometric_mean_filter_color(image, kernel_size=3):
    b, g, r = cv2.split(image)
    b_filtered = geometric_mean_filter(b, kernel_size)
    g_filtered = geometric_mean_filter(g, kernel_size)
    r_filtered = geometric_mean_filter(r, kernel_size)
    filtered_image = cv2.merge([b_filtered, g_filtered, r_filtered])
    clean_image = Image.fromarray(filtered_image)
    clean_tensor = preprocess(clean_image).unsqueeze(0)
    with torch.no_grad():
        output = googlenet(clean_tensor)
        predicted_idx = output.argmax().item()
    return labels[predicted_idx]

def gaussian_blur_denoising_algorithm(image):
    clean_image = cv2.GaussianBlur(image, (15, 15), 0)
    clean_image = Image.fromarray(clean_image)
    clean_tensor = preprocess(clean_image).unsqueeze(0)
    with torch.no_grad():
        output = googlenet(clean_tensor)
        predicted_idx = output.argmax().item()
    return labels[predicted_idx]

def alpha_trimmed_mean_filter(image, kernel_size=3, alpha=2):
    H, W, C = image.shape
    pad = kernel_size // 2
    filtered_image = np.zeros_like(image, dtype=np.float32)
    for ch in range(C):
        padded_channel = cv2.copyMakeBorder(image[:, :, ch], pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        for i in range(H):
            for j in range(W):
                region = padded_channel[i:i + kernel_size, j:j + kernel_size].flatten()
                region_sorted = np.sort(region)
                trimmed_region = region_sorted[alpha:-alpha] if alpha > 0 else region_sorted
                filtered_image[i, j, ch] = np.mean(trimmed_region)
    filtered_image = filtered_image.astype(np.uint8)
    clean_image = Image.fromarray(filtered_image)
    clean_tensor = preprocess(clean_image).unsqueeze(0)
    with torch.no_grad():
        output = googlenet(clean_tensor)
        predicted_idx = output.argmax().item()
    return labels[predicted_idx]

# Object label before denoising
def object_label_before_denoising_func(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = googlenet(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
    return labels[predicted_idx.item()], confidence.item() * 100

# Denoising stack
def denoising_stack(image, kernel_size=3, alpha=2):
    outputs = []
    outputs.append(fast_non_local_means_denoising_algorithm(image))
    outputs.append(midpoint_filter(image, kernel_size))
    outputs.append(bilateral_filtering_algorithm(image))
    outputs.append(geometric_mean_filter_color(image, kernel_size))
    outputs.append(gaussian_blur_denoising_algorithm(image))
    outputs.append(alpha_trimmed_mean_filter(image, kernel_size, alpha))
    return outputs

# Majority voting
def majority_voting(denoising_stack_output):
    if not denoising_stack_output:
        return None
    counter = Counter(denoising_stack_output)
    return counter.most_common(1)[0][0]

# Adversarial object detection
def adversarial_object_detection(label_before, label_after):
    return "Clean Object Detected!!!" if label_before == label_after else "Adversarial Object Detected!!!"

def image_enhancer():
    st.markdown("""
        <style>
        .result-box {
            background-color: rgba(0, 0, 0, 0.7);
            border: 2px solid #00f0ff;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            color: white;
        }
        .subheader {
            color: white;
            font-size: 24px;
            font-weight: bold;
            text-shadow: 0 0 5px #00f0ff;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="glow" style="color:white;">Camera Spoofing Adversarial Attacks Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 20px; color: white;">Enhancing CNN Robustness with Spatial-Domain Denoising in Self-Driving Systems</p>', unsafe_allow_html=True)

    # Sidebar for parameters
    with st.sidebar:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown('<div class="feature-title">Configuration</div>', unsafe_allow_html=True)
        kernel_size = st.slider("Kernel Size for Filters", min_value=3, max_value=7, value=3, step=2)
        alpha = st.slider("Alpha for Alpha-Trimmed Mean Filter", min_value=0, max_value=4, value=2)
        st.markdown('</div>', unsafe_allow_html=True)

    # Image upload
    st.markdown('<div class="subheader">Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image (JPG, PNG)", type=["jpg", "jpeg", "png"], key="enhancer_uploader")

    if uploaded_file is not None:
        # Generate unique artifact ID
        artifact_id = str(uuid.uuid4())

        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        # Convert to OpenCV format
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Run pipeline
        with st.spinner("Processing image..."):
            label_before, confidence_before = object_label_before_denoising_func(image_cv)
            denoising_outputs = denoising_stack(image_cv, kernel_size, alpha)
            label_after = majority_voting(denoising_outputs)
            final_status = adversarial_object_detection(label_before, label_after)

        # Display results
        st.markdown('<div class="subheader">Results</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="result-box" >', unsafe_allow_html=True)
            st.markdown('<b style="color:white;">Prediction Before Denoising</b>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:white;"><b>Class</b>: {label_before}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:white;"><b>Confidence</b>: {confidence_before:.2f}%</p>', unsafe_allow_html=True)
            #st.write(f"**Class**: {label_before}")
            #st.write(f"**Confidence**: {confidence_before:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown('<b style="color:white;">Denoising Stack Predictions</b>', unsafe_allow_html=True)
            algorithms = [
                "Fast Non-Local Means",
                "Midpoint Filter",
                "Bilateral Filter",
                "Geometric Mean Filter",
                "Gaussian Blur",
                "Alpha-Trimmed Mean"
            ]
            for algo, output in zip(algorithms, denoising_outputs):
                st.markdown(f'<p style="color:white;"><b>{algo}</b>: {output}</p>', unsafe_allow_html=True)
                #st.write(f"**{algo}**: {output}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown('<b style="color:white;">Majority Voted Prediction</b>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:white;"><b>Class</b>: {label_after}</p>', unsafe_allow_html=True)
        #st.write(f"**Class**: {label_after}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown('<b style="color:white;">Final Classification</b>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:white;"><b>Status</b>: {final_status}</p>', unsafe_allow_html=True)
        #st.markdown(f"**Status**: {final_status}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<p style="display:none">Artifact ID: {artifact_id}</p>', unsafe_allow_html=True)

    else:
        st.markdown('<p style="color: white;">Please upload an image to start the classification.</p>', unsafe_allow_html=True)

    if st.button("🔙 Back to Home"):
        st.session_state['nav'] = "Home"


###------------------------------------------------------------------------------------------------------------------------------

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
    Lidar_Spoofing_Detection()
elif app_mode == "Ransika":
    image_enhancer()
