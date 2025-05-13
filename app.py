# import streamlit as st
# import base64

# # Set page layout and title
# st.set_page_config(
#     page_title="GALACTIC-X",
#     layout="wide",
#     page_icon="gs-page-logo.png",
# )

# # Function to add background image (using base64 encoding)
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as f:
#         encoded_string = base64.b64encode(f.read()).decode()
#     css = f"""
#     <style>
#     .stApp {{
#         background-image: url("data:image/jpg;base64,{encoded_string}");
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#     }}
#     </style>
#     """
#     st.markdown(css, unsafe_allow_html=True)

# # Function to display the home page
# # def home_page():
# #     glowing_title = """
# #     <style>
# #     .glow {
# #         font-size: 100px;
# #         font-weight: bold;
# #         text-align: center;
# #         color: white;
# #         text-shadow: 0 0 9px white, 0 0 20px #00004B, 
# #                      0 0 20px #00004B, 0 0 20px #00004B;
# #         margin-bottom: 0px;
# #     }
# #     </style>
# #     """
# #     st.markdown(glowing_title, unsafe_allow_html=True)
# #     st.markdown('<h1 class="glow">GALACTIC X</h1>', unsafe_allow_html=True)

# #     st.markdown(
# #         """<p style="text-align: center; font-size: 30px;">
# #         <b>Welcome to GALACTIC-X! Your all-in-one tool for advanced galaxy data analysis.</b></p>""",
# #         unsafe_allow_html=True,
# #     )

# #     st.markdown(
# #         """<h1 style="text-align: center; font-size: 50px;">What do we Provide?</h1>""",
# #         unsafe_allow_html=True,
# #     )

# #     col1, col2, col3, col4, col5 = st.columns(5)

# #     def info_box(col, icon, text):
# #         with col:
# #             with st.container(border=True, height=400):
# #                 _, center, _, _, _ = st.columns(5)
# #                 with center:
# #                     st.image(icon, width=30)
# #                 st.markdown(f"""
# #                 <div style="text-align: center; font-weight: bold;">
# #                     <p style="font-size:20px"><b>{text}</b></p>
# #                 </div>
# #                 """, unsafe_allow_html=True)

# #     info_box(col1, "classification.png", "Dynamically predict galaxy types, redshift values, or star formation rates using machine learning.")
# #     info_box(col2, "clustering.png", "Utilize machine learning algorithms to classify galaxies based on morphology, brightness, and color.")
# #     info_box(col3, "globular.png", "Detect and analyze globular clusters using photometric and classification algorithms.")
# #     info_box(col4, "enhancer.png", "Use GANs to enhance low-resolution galaxy images for better analysis.")
# #     info_box(col5, "chatbot.png", "Ask Astronerd any astronomy-related questions with instant, accurate answers.")

# # Only load the home page (no sidebar)
# def home_page():
#     # Inject CSS for hover effect
#     st.markdown("""
#         <style>
#         .feature-box {
#             background-color: rgba(0, 0, 0, 0.5);
#             border: 2px solid transparent;
#             border-radius: 15px;
#             padding: 20px;
#             text-align: center;
#             transition: all 0.3s ease;
#             height: 100%;
#         }

#         .feature-box:hover {
#             border: 2px solid #00f0ff;
#             box-shadow: 0 0 20px #00f0ff;
#             transform: scale(1.03);
#             cursor: pointer;
#         }

#         .feature-title {
#             font-size: 20px;
#             font-weight: bold;
#             color: white;
#         }

#         .feature-desc {
#             font-size: 16px;
#             color: white;
#         }

#         </style>
#     """, unsafe_allow_html=True)

#     st.markdown('<h1 class="glow">GALACTIC X</h1>', unsafe_allow_html=True)
#     st.markdown(
#         """
#         <p style="text-align: center; font-size: 30px;"><b>Welcome to GALACTIC-X! Your all-in-one tool for advanced galaxy data analysis.</b></p>
#         """,
#         unsafe_allow_html=True,
#     )

#     st.markdown(
#         """
#         <h1 style="text-align: center; font-size: 50px;">What do we Provide?</h1>
#         """,
#         unsafe_allow_html=True,
#     )

#     col1, col2, col3, col4 = st.columns(4)

#     with col1:
#         with st.container():
#             st.markdown('<div class="feature-box">', unsafe_allow_html=True)
#             st.image("classification.png", width=60)
#             st.markdown('<div class="feature-title">Galaxy Classification</div>', unsafe_allow_html=True)
#             st.markdown('<div class="feature-desc">Predict galaxy types and redshifts with ML models.</div>', unsafe_allow_html=True)
#             if st.button("Explore Classifier", key="classifier"):
#                 st.session_state['nav'] = "Romold"
#             st.markdown('</div>', unsafe_allow_html=True)

#     with col2:
#         with st.container():
#             st.markdown('<div class="feature-box">', unsafe_allow_html=True)
#             st.image("clustering.png", width=60)
#             st.markdown('<div class="feature-title">Cluster Analysis</div>', unsafe_allow_html=True)
#             st.markdown('<div class="feature-desc">Analyze spatial clustering of galaxies and clusters.</div>', unsafe_allow_html=True)
#             if st.button("Explore Clustering", key="clustering"):
#                 st.session_state['nav'] = "Tharindu"
#             st.markdown('</div>', unsafe_allow_html=True)

#     with col3:
#         with st.container():
#             st.markdown('<div class="feature-box">', unsafe_allow_html=True)
#             st.image("globular.png", width=60)
#             st.markdown('<div class="feature-title">Globular Analysis</div>', unsafe_allow_html=True)
#             st.markdown('<div class="feature-desc">Detect and study globular clusters in galaxies.</div>', unsafe_allow_html=True)
#             if st.button("Explore Globulars", key="globular"):
#                 st.session_state['nav'] = "Pasindu"
#             st.markdown('</div>', unsafe_allow_html=True)

#     with col4:
#         with st.container():
#             st.markdown('<div class="feature-box">', unsafe_allow_html=True)
#             st.image("enhancer.png", width=60)
#             st.markdown('<div class="feature-title">Image Enhancer</div>', unsafe_allow_html=True)
#             st.markdown('<div class="feature-desc">Upscale galaxy images using deep learning GANs.</div>', unsafe_allow_html=True)
#             if st.button("Explore Enhancer", key="enhancer"):
#                 st.session_state['nav'] = "Ransika"
#             st.markdown('</div>', unsafe_allow_html=True)

# add_bg_from_local("bg-7.jpg")
# home_page()

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
import pickle

# Set page config
st.set_page_config(
    page_title="Nirbhaya",
    layout="wide",
    page_icon="gs-page-logo.png",
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

# Dummy feature pages (replace with your actual content)


def ddos_classifier():
    st.title("DDoS Detection System 🚀")
    st.write("Provide the input values for the **Top 10 SHAP Features** below:")

    # Load saved model and scaler
    model = models.load_model("models/transformer_top10_model.h5")
    with open(models_dir/"scaler_ddos.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(models_dir/"top10_features.pkl", "rb") as f:
        top_10_feature_names = pickle.load(f)

    # User Inputs for 10 features
    user_input = {}
    for feature in top_10_feature_names:
        user_input[feature] = st.number_input(f"{feature}", value=0.0)

    # Predict button
    if st.button("🚨 Predict DDoS Attack"):
        # Convert to array
        input_values = np.array([list(user_input.values())])

        # Scale and reshape
        input_scaled = scaler.transform(input_values)
        input_reshaped = np.expand_dims(input_scaled, axis=2)

        # Predict
        prediction = model.predict(input_reshaped)[0][0]

        if prediction >= 0.5:
            st.error(f"⚠️ DDoS Attack Detected! Probability: {prediction:.2f}")
        else:
            st.success(f"✅ Normal Traffic. Probability: {1 - prediction:.2f}")

    # Navigation
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
add_bg_from_local("bg-7.jpg")
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
