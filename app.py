# ==========================================
# app.py
# AI-Based Mental Health Sentiment Monitoring
# Advanced UI Version
# ==========================================

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
import numpy as np
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1f4e79;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    color: #5c677d;
    margin-bottom: 30px;
}

.section-card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    margin-bottom: 25px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model(
    "mental_health_rnn_model.h5"
)

# ==========================================
# LOAD TOKENIZER
# ==========================================

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# ==========================================
# LOAD LABEL ENCODER
# ==========================================

with open("label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

# ==========================================
# PARAMETERS
# ==========================================

max_length = 80

# ==========================================
# PREPROCESSING FUNCTION
# ==========================================

def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    return text

# ==========================================
# EMOTIONAL GUIDANCE
# ==========================================

def emotional_guidance(emotion):

    guidance = {

        "Anxiety":
        "Take deep breaths and practice grounding exercises. Try talking with someone you trust.",

        "Depression":
        "Your feelings are important. Reach out to supportive people and avoid isolating yourself.",

        "Stress":
        "Take short breaks, stay hydrated, and divide tasks into smaller goals.",

        "Suicidal":
        "Please contact someone immediately. You are not alone and support is available.",

        "Bipolar":
        "Maintain a stable routine and monitor emotional fluctuations carefully.",

        "Personality disorder":
        "Self-awareness and emotional regulation practices may help improve balance.",

        "Normal":
        "Keep maintaining a healthy and positive lifestyle."
    }

    return guidance.get(
        emotion,
        "Take care of your emotional well-being."
    )

# ==========================================
# HEADER SECTION
# ==========================================

st.markdown(
    """
    <div class='title'>
    🧠 AI-Based Mental Health Sentiment Monitoring System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='subtitle'>
    Emotion Detection using Advanced Bidirectional Simple RNN
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ==========================================
# ABOUT PROJECT SECTION
# ==========================================

st.markdown(
    """
    <div class='section-card'>
    <h2>📘 About the Project</h2>

    <p>
    This project uses Artificial Intelligence and Natural Language Processing (NLP)
    to analyze emotional sentiment from user text messages.
    </p>

    <p>
    The system helps identify emotions such as:
    Anxiety, Depression, Stress, Suicidal thoughts,
    Bipolar disorder, Personality disorders, and Normal emotional states.
    </p>

    <p>
    Bidirectional Simple Recurrent Neural Networks (Bi-SimpleRNN)
    are used to understand sequential emotional patterns
    from text data.
    </p>

    <h4>Applications:</h4>

    <ul>
        <li>Mental Health Monitoring</li>
        <li>AI-based Emotional Analysis</li>
        <li>Healthcare NLP Systems</li>
        <li>Virtual Counseling Assistants</li>
        <li>Emotion-aware AI Systems</li>
    </ul>

    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# USER INPUT SECTION
# ==========================================

st.markdown(
    """
    <div class='section-card'>
    <h2>✍️ Enter Your Thoughts or Feelings</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.info("Example: I feel emotionally tired and stressed today.")

st.info("Example: Nobody understands me and I feel lonely.")

st.info("Example: I am very happy and excited about my future.")

user_input = st.text_area(
    "Enter your thoughts or feelings here...",
    height=220
)

# ==========================================
# PREDICTION BUTTON
# ==========================================

analyze = st.button(
    "🔍 Analyze Emotion"
)

# ==========================================
# PREDICTION LOGIC
# ==========================================

if analyze:

    if user_input.strip() == "":

        st.warning(
            "Please enter some text."
        )

    else:

        # ==================================
        # PREPROCESS
        # ==================================

        processed_text = preprocess_text(
            user_input
        )

        # ==================================
        # TOKENIZATION
        # ==================================

        sequence = tokenizer.texts_to_sequences(
            [processed_text]
        )

        # ==================================
        # PADDING
        # ==================================

        padded = pad_sequences(
            sequence,
            maxlen=max_length,
            padding='post'
        )

        # ==================================
        # MODEL PREDICTION
        # ==================================

        prediction = model.predict(
            padded
        )

        predicted_index = np.argmax(
            prediction
        )

        confidence = np.max(
            prediction
        )

        predicted_emotion = encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence_percentage = round(
            confidence * 100,
            2
        )

        # ==================================
        # OUTPUT SECTION
        # ==================================

        st.markdown(
            """
            <div class='section-card'>
            <h2>📊 Prediction Output</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                f"Emotion Detected: {predicted_emotion}"
            )

        with col2:

            st.info(
                f"Confidence Score: {confidence_percentage}%"
            )

        # ==================================
        # EMOTIONAL STATUS
        # ==================================

        if predicted_emotion in [
            "Depression",
            "Stress",
            "Anxiety",
            "Suicidal"
        ]:

            st.error(
                "⚠️ Emotional Status: Needs Emotional Attention"
            )

        else:

            st.success(
                "✅ Emotional Status: Emotionally Stable"
            )

        # ==================================
        # VISUALIZATION SECTION
        # ==================================

        st.markdown(
            """
            <div class='section-card'>
            <h2>📈 Sentiment Confidence Visualization</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        emotions = encoder.classes_

        probabilities = prediction[0]

        chart_data = pd.DataFrame({
            "Emotion": emotions,
            "Confidence": probabilities
        })

        st.bar_chart(
            chart_data.set_index("Emotion")
        )

        # ==================================
        # DETAILED PROBABILITY TABLE
        # ==================================

        st.subheader(
            "Detailed Confidence Scores"
        )

        chart_data["Confidence"] = (
            chart_data["Confidence"] * 100
        ).round(2)

        st.dataframe(
            chart_data,
            use_container_width=True
        )

        # ==================================
        # WELLNESS GUIDANCE
        # ==================================

        st.markdown(
            """
            <div class='section-card'>
            <h2>💙 Emotional Wellness Guidance</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        guidance_message = emotional_guidance(
            predicted_emotion
        )

        st.success(
            guidance_message
        )

        st.markdown("""
        ### 🌟 Positive Wellness Tips

        ✅ Stay hydrated  
        ✅ Practice mindfulness  
        ✅ Take regular breaks  
        ✅ Sleep properly  
        ✅ Talk with trusted people  
        ✅ Exercise regularly  
        ✅ Maintain healthy routines  
        ✅ Seek professional support if needed
        """)

# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <div class='footer'>
    Developed using Streamlit, TensorFlow, NLP, and Bidirectional SimpleRNN
    </div>
    """,
    unsafe_allow_html=True
)