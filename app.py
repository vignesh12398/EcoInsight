from fileinput import filename

import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import pandas  as pd
import os
from datetime import datetime as dt

@st.cache_resource
def load_models():
  waste_model = load_model("model/mobilenet_finetuned_best.keras")
  ml = joblib.load("model/carbon_model.pkl")
  encoders = joblib.load("model/encoders.pkl")
  return waste_model, ml, encoders
waste_model, ml, encoders = load_models()
st.set_page_config(page_title="Carbon Footprint Predictor",
                   page_icon="🌍",
                   layout="wide")

st.markdown("""
<style>
    /* Overall dark background */
    .stApp {
        background: radial-gradient(circle at 20% 0%, #14201a 0%, #0e1512 45%, #0a0f0d 100%);
        color: #e6efe9;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f1713;
        border-right: 1px solid #1f2e26;
    }
    section[data-testid="stSidebar"] * {
        color: #cfe3d7 !important;
    }

    /* Default text elements */
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #e6efe9;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #123d24 0%, #1c6b3d 60%, #2c9556 100%);
        padding: 2.2rem 2.2rem 1.8rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
        border: 1px solid #245c37;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        color: white !important;
    }
    .hero p {
        margin-top: 0.4rem;
        font-size: 1.05rem;
        opacity: 0.92;
        color: #e8f5ec !important;
    }

    /* Section cards */
    .section-card {
        background: #131e18;
        border-radius: 16px;
        padding: 1.4rem 1.6rem 0.6rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.35);
        border: 1px solid #223a2b;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #4fd280 !important;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Inputs */
    div[data-baseweb="select"] > div {
        background-color: #17241c;
        border-color: #2a4232;
        cursor: pointer;
    }
    div[data-baseweb="select"] * { cursor: pointer !important; }
    input, textarea {
        background-color: #17241c !important;
        color: #e6efe9 !important;
    }
    div[data-testid="stNumberInput"] input {
        background-color: #17241c !important;
        color: #e6efe9 !important;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1c6b3d 0%, #2fa15c 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.7rem 0;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(47, 161, 92, 0.35);
    }

    /* Result card */
    .result-card {
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 6px 24px rgba(0,0,0,0.45);
        border: 1px solid rgba(255,255,255,0.06);
    }
    .result-value {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }
    .result-label {
        font-size: 1rem;
        opacity: 0.9;
    }

    .tip-item {
        background: #131e18;
        border-left: 4px solid #2fa15c;
        padding: 0.6rem 0.9rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        color: #dcecdf;
    }

    hr, div[data-testid="stDivider"] {
        border-color: #223a2b !important;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# Hero Header
# ===========================


# ===========================
# Sidebar
# ===========================
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This tool predicts your estimated **monthly carbon emissions** "
        "(in kg CO₂) based on your transport habits, energy use, and lifestyle choices."
    )
    st.divider()
    st.markdown("**Emission bands**")
    st.markdown("🟢 Low — under 1,500 kg CO₂")
    st.markdown("🟡 Moderate — 1,500–2,500 kg CO₂")
    st.markdown("🔴 High — above 2,500 kg CO₂")
    st.divider()
    st.caption("Built with Streamlit • Powered by your trained ML model")

# ===========================
# User Inputs
# ===========================
page = st.sidebar.radio(
    "Go to",
        [
                 "🏠 Home",
                 "🌍 Carbon Footprint Calculator",
                  "♻️ Waste Classification",
                    "📊 Feedback (Admin)"


        ],
         index=0
)
if page == "🏠 Home":
    # Hero section
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <h1 style="font-size:2.8rem; margin-bottom:0;">🌱 CarbonAI</h1>
            <p style="font-size:1.2rem; color:#6b7280; margin-top:5px;">
                AI-Powered Sustainability Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        CarbonAI combines **Machine Learning** and **Deep Learning** to help you understand
        your environmental impact and make smarter, greener decisions — one prediction,
        one image, one habit at a time.
        """
    )

    st.divider()

    # Quick stats / value props
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Modules Live", value="2")
    with col2:
        st.metric(label="ML Model", value="Regression")
    with col3:
        st.metric(label="DL Model", value="MobileNetV2")

    st.divider()

    # Feature cards
    st.subheader("🚀 What CarbonAI Does")

    feat_col1, feat_col2 = st.columns(2)

    with feat_col1:
        st.markdown(
            """
            <div style="background:#131e18; padding:20px; border-radius:12px; 
                        border:1px solid #2fa15c; height:230px;">
                <h4 style="color:#4fd280;">🌍 Carbon Footprint Calculator</h4>
                <p style="color:#dcecdf;">Predicts your carbon emissions based on lifestyle, transport,
                and energy usage patterns using a trained ML regression model.</p>
                <p style="color:#dcecdf;"><b>✔ Instant emission estimate</b><br>
                <b>✔ Personalized reduction insights</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with feat_col2:
        st.markdown(
            """
            <div style="background:#131e2a; padding:20px; border-radius:12px; 
                        border:1px solid #3f7fd6; height:230px;">
                <h4 style="color:#7cb3ff;">♻️ AI Waste Classification</h4>
                <p style="color:#dce6f0;">Upload an image of waste and our fine-tuned MobileNetV2 model will classify it into one of six categories:
**Cardboard, Glass, Metal, Organic, Paper, or Plastic.*</p>
                <p style="color:#dce6f0;"><b>✔ Image-based detection</b><br>
                <b>✔ Encourages correct disposal</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # Tech stack
    st.subheader("🛠 Technologies Used")
    tech_col1, tech_col2, tech_col3, tech_col4, tech_col5 = st.columns(5)
    for col, tech in zip(
        [tech_col1, tech_col2, tech_col3, tech_col4, tech_col5],
        ["🐍 Python", "🎈 Streamlit", "🔢 Scikit-learn", "🧠 TensorFlow/Keras", "📊 Pandas & NumPy"]
    ):
        col.markdown(f"<div style='text-align:center; font-size:0.9rem;'>{tech}</div>", unsafe_allow_html=True)

    st.divider()

    # Objective
    st.subheader("🎯 Our Objective")
    st.info(
        "To encourage sustainable practices by helping users understand their carbon "
        "footprint and dispose of waste responsibly — powered by AI, driven by impact."
    )

    st.success("👈 Select a module from the sidebar to get started.")

if page=="🌍 Carbon Footprint Calculator":
    st.markdown("""
    <div class="hero">
        <h1>🌍 Carbon Footprint Prediction System</h1>
        <p>Estimate your monthly carbon emissions using Machine Learning, and get personalized sustainability tips.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-card"><div class="section-title">🚗 Transport</div>', unsafe_allow_html=True)
    transport = st.selectbox(
        "Transport",
        encoders["Transport"].classes_
    )

    vehicle_type = st.selectbox(
        "Vehicle Type",
        encoders["Vehicle Type"].classes_
    )

    distance = st.number_input(
        "Vehicle Monthly Distance (km)",
        min_value=0,
        value=100,
        step=10
    )
    flight = st.selectbox(
        "Frequency of Air Travel",
        encoders["Frequency of Traveling by Air"].classes_
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">⚡ Energy</div>', unsafe_allow_html=True)
    energy = st.selectbox(
        "Energy Efficiency",
        encoders["Energy efficiency"].classes_
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">♻️ Waste & Recycling</div>', unsafe_allow_html=True)
    recycling = st.multiselect(
        "Materials You Recycle",
        ["Paper", "Plastic", "Glass", "Metal"]
    )
    waste = st.slider(
        "Waste Bags per Week",
        min_value=1,
        max_value=7,
        value=3
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">🛒 Consumption</div>', unsafe_allow_html=True)
    grocery = st.number_input(
        "Monthly Grocery Bill",
        min_value=0,
        value=1000,
        step=100
    )
    st.markdown('</div>', unsafe_allow_html=True)



    predict = st.button("🔍 Predict Carbon Emission")
    if predict:
        transport = encoders["Transport"].transform([transport])[0]

        vehicle_type = encoders["Vehicle Type"].transform([vehicle_type])[0]

        energy = encoders["Energy efficiency"].transform([energy])[0]
        recycling = str(recycling)
        recycling = encoders["Recycling"].transform([recycling])[0]

        flight = encoders["Frequency of Traveling by Air"].transform([flight])[0]
        features = [[
            transport,
            vehicle_type,
            distance,
            energy,
            waste,
            recycling,
            grocery,
            flight
        ]]
        prediction = ml.predict(features)[0]
        if prediction < 1500:
            bg = "linear-gradient(135deg, #123d24 0%, #1c6b3d 100%)"
            text_color = "#7cf0a8"
            emoji = "🟢"
            label = "Low Emission"
        elif prediction < 2500:
            bg = "linear-gradient(135deg, #4a3b0f 0%, #7a5f14 100%)"
            text_color = "#ffd76a"
            emoji = "🟡"
            label = "Moderate Emission"
        else:
            bg = "linear-gradient(135deg, #4a1414 0%, #7a1f1f 100%)"
            text_color = "#ff8787"
            emoji = "🔴"
            label = "High Emission"

        st.markdown(f"""
            <div class="result-card" style="background:{bg};">
                <div class="result-label" style="color:{text_color};">{emoji} {label}</div>
                <div class="result-value" style="color:{text_color};">{prediction:.2f} kg CO₂</div>
                <div class="result-label" style="color:{text_color};">Estimated monthly carbon emission</div>
            </div>
            """, unsafe_allow_html=True)
if page == "♻️ Waste Classification":
    st.markdown("""
        <style>
        .upload-header {
            font-size: 26px;
            font-weight: 700;
            color: #2E7D32 !important;
            margin-bottom: 5px;
        }
        .sub-text {
            color: #cccccc !important;
            font-size: 15px;
            margin-bottom: 20px;
        }
        .result-card {
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    margin-top: 10px;
    text-align: center;
    transition: transform 0.2s ease;
}

        .result-card:hover {
            transform: translateY(-3px);
        }
        
        .result-title {
            font-size: 26px;
            font-weight: 700;
            color: #212121;
            margin-bottom: 8px;
        }
        
        .confidence-text {
            font-size: 16px;
            font-weight: 500;
            color: #424242;
        }
        
        .organic-card {
            background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
            border: 2px solid #4CAF50;
        }
        
        .cardboard-card {
            background: linear-gradient(135deg, #EFEBE9, #D7CCC8);
            border: 2px solid #795548;
        }
        
        .metal-card {
            background: linear-gradient(135deg, #ECEFF1, #CFD8DC);
            border: 2px solid #607D8B;
        }
        
        .plastic-card {
            background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
            border: 2px solid #2196F3;
        }
        
        .paper-card {
            background: linear-gradient(135deg, #F5F5F5, #E0E0E0);
            border: 2px solid #9E9E9E;
        }
        
        .glass-card {
            background: linear-gradient(135deg, #E0F7FA, #B2EBF2);
            border: 2px solid #00BCD4;
        }
        
        .default-card {
            background: linear-gradient(135deg, #FAFAFA, #E0E0E0);
            border: 2px solid #9E9E9E;
        }
               
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="upload-header">♻️ Waste Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Upload an image to check if it\'s Organic or Recyclable waste.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a waste image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            img = image.load_img(uploaded_file, target_size=(224, 224))
            st.image(img, caption="Uploaded Image", use_container_width=True)

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        with st.spinner("Analyzing image..."):
            predictions = waste_model.predict(img_array)



        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        class_labels = ['cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic']
        label = class_labels[predicted_class]


        icon_map = {
            "Organic": "🌱",
            "Cardboard": "📦",
            "Metal": "🔩",
            "Plastic": "🧴",
            "Paper": "📄",
            "Glass": "🍶",
        }

        card_class_map = {
            "Organic": "organic-card",
            "Cardboard": "cardboard-card",
            "Metal": "metal-card",
            "Plastic": "plastic-card",
            "Paper": "paper-card",
            "Glass": "glass-card",
        }

        icon = icon_map.get(label, "♻️")
        card_class = card_class_map.get(label, "default-card")

        with col2:
            st.markdown(f"""
                <div class="result-card {card_class}">
                    <div class="result-title">{icon} {label}</div>
                    <div class="confidence-text">Confidence: {confidence:.2f}%</div>
                      <hr style="margin:12px 0;">

            

            </div>
            """, unsafe_allow_html=True)

            st.progress(min(int(confidence), 100))

            st.markdown("""
                <style>
                .feedback-question {
                    font-size: 22px;
                    font-weight: 700;
                    color: #ffffff;
                    margin-top: 25px;
                    margin-bottom: 20px;
                }
                div[data-testid="stButton"] {
                    width: 100% !important;
                }
                div[data-testid="stButton"] > button {
                    width: 100% !important;
                    min-height: 70px !important;
                    border-radius: 14px !important;
                    padding: 10px 20px !important;
                    font-weight: 700 !important;
                    font-size: 20px !important;
                    letter-spacing: 0.3px !important;
                    border: none !important;
                    color: #ffffff !important;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
                    transition: all 0.2s ease-in-out !important;
                }
                div[data-testid="stButton"] > button p {
                    font-size: 20px !important;
                    font-weight: 700 !important;
                }
                /* Yes button */
                #yes-btn-marker + div div[data-testid="stButton"] > button {
                    background: linear-gradient(180deg, #34a869 0%, #1f7a4a 100%) !important;
                }
                #yes-btn-marker + div div[data-testid="stButton"] > button:hover {
                    background: linear-gradient(180deg, #3fc47b 0%, #248f56 100%) !important;
                    transform: translateY(-3px);
                }
                /* No button */
                #no-btn-marker + div div[data-testid="stButton"] > button {
                    background: linear-gradient(180deg, #d9534f 0%, #a83732 100%) !important;
                }
                #no-btn-marker + div div[data-testid="stButton"] > button:hover {
                    background: linear-gradient(180deg, #e56864 0%, #bb433d 100%) !important;
                    transform: translateY(-3px);
                }
                /* Submit Feedback button */
                #submit-btn-marker + div div[data-testid="stButton"] > button {
                    background: linear-gradient(180deg, #4b7bd1 0%, #2d54a3 100%) !important;
                }
                #submit-btn-marker + div div[data-testid="stButton"] > button:hover {
                    background: linear-gradient(180deg, #5c8ce0 0%, #3862b8 100%) !important;
                    transform: translateY(-3px);
                }
                label[data-testid="stWidgetLabel"] p {
                    color: #d0d0d0 !important;
                    font-size: 16px;
                    font-weight: 600;
                }
                div[data-baseweb="select"] > div {
                    background-color: #1a1a1a;
                    border: 1px solid #3a3a3a;
                    border-radius: 10px;
                    color: #ffffff;
                }
                div[data-baseweb="select"] > div:hover {
                    border-color: #4caf7d;
                }
                </style>
            """, unsafe_allow_html=True)

            st.markdown('<p class="feedback-question">Was the prediction right?</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                if(st.button("👍 Yes")):
                    feedback = {
                        "Timestamp": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Prediction Class": label,
                        "Correct Class": label,  # Same as prediction
                        "Confidence": confidence
                    }

                    file_name = "feedback.csv"

                    if os.path.exists(file_name):
                        df = pd.read_csv(file_name)
                        df = pd.concat([df, pd.DataFrame([feedback])], ignore_index=True)
                    else:
                        df = pd.DataFrame([feedback])

                    df.to_csv(file_name, index=False)

                    st.success("✅ Thank you! Your feedback has been saved.")

            if "show_feedback" not in st.session_state:
                st.session_state.show_feedback = False

            with col2:
                if st.button("👎 No"):
                    st.session_state.show_feedback = True

            if st.session_state.show_feedback:

                st.write("Select the correct class")

                correct_class = st.selectbox(
                    "Correct Class",
                    ["cardboard", "paper", "plastic", "metal", "organic", "glass"]
                )

                if st.button("Submit Feedback"):

                    feedback = {
                        "Timestamp": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Prediction Class": label,
                        "Correct Class": correct_class,
                        "Confidence": confidence
                    }

                    file_name = "feedback.csv"

                    if os.path.exists(file_name):
                        df = pd.read_csv(file_name)
                        df = pd.concat([df, pd.DataFrame([feedback])], ignore_index=True)
                    else:
                        df = pd.DataFrame([feedback])

                    df.to_csv(file_name, index=False)

                    st.success("✅ Feedback saved!")

                    st.session_state.show_feedback = False

if page=="📊 Feedback (Admin)":
    st.markdown("""
            <style>
            .feedback-header {
                background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
                padding: 20px 25px;
                border-radius: 12px;
                margin-bottom: 25px;
            }
            .feedback-header h1 {
                color: white;
                font-size: 28px;
                margin: 0;
            }
            div[data-testid="stMetric"] {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                border-radius: 12px;
                padding: 18px 15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            }
            div[data-testid="stMetricLabel"] {
                font-weight: 600;
                color: #b0b0b0 !important;
            }
            div[data-testid="stMetricLabel"] p {
                color: #b0b0b0 !important;
            }
            div[data-testid="stMetricValue"] {
                font-size: 30px;
                font-weight: 700;
                color: #ffffff !important;
            }
            div[data-testid="stDataFrame"] {
                border: 1px solid #333333;
                border-radius: 10px;
                overflow: hidden;
            }
            </style>
            <div class="feedback-header">
                <h1>📊 Feedback (Admin)</h1>
            </div>
        """, unsafe_allow_html=True)
    df=pd.read_csv("feedback.csv")
    total_feedback=len(df)
    correct_predicted=(df["Prediction Class"] == df["Correct Class"]).sum()
    wrong_predicted=total_feedback-correct_predicted
    col1,col2,col3 = st.columns(3)
    col1.metric("Total Feedback",total_feedback)
    col2.metric("Correct Prediction",correct_predicted)
    col3.metric("Wrong Prediction",wrong_predicted)
    st.divider()
    st.dataframe(df,use_container_width=True)
