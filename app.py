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

# Set page config
st.set_page_config(
    page_title="Nirbhaya",
    layout="wide",
    page_icon="gs-page-logo.png",
)

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

    st.markdown('<h1 class="glow">GALACTIC X</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 30px;"><b>Welcome to GALACTIC-X! Your all-in-one tool for advanced galaxy data analysis.</b></p>', unsafe_allow_html=True)
    st.markdown('<h1 style="text-align: center; font-size: 50px;">What do we Provide?</h1>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container():
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.image("classification.png", width=60)
            st.markdown('<div class="feature-title">DDoS Classification</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-desc">Predict galaxy types and redshifts with ML models.</div>', unsafe_allow_html=True)
            if st.button("Explore Classifier", key="classifier"):
                st.session_state['nav'] = "Romold"
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.image("clustering.png", width=60)
            st.markdown('<div class="feature-title">Tharindu Component</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-desc">Analyze spatial clustering of galaxies and clusters.</div>', unsafe_allow_html=True)
            if st.button("Explore Clustering", key="clustering"):
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
def galaxy_classifier():
    st.title("DDoS")
    st.write("🚀 This is a placeholder for the galaxy classification page.")
    if st.button("🔙 Back to Home"):
        st.session_state['nav'] = "Home"

def galaxy_clustering():
    st.title("Tharindus")
    st.write("🧠 This is a placeholder for the clustering page.")
    if st.button("🔙 Back to Home"):
        st.session_state['nav'] = "Home"

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
    galaxy_classifier()
elif app_mode == "Tharindu":
    galaxy_clustering()
elif app_mode == "Pasindu":
    globular_clusters()
elif app_mode == "Ransika":
    image_enhancer()
