# ==========================================
# STREAMLIT APP
# AI-Based Mental Health Sentiment Monitoring
# ==========================================

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    layout="wide"
)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = load_model("mental_health_rnn_model.h5")

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

max_length = 50

# ==========================================
# TEXT PREPROCESSING FUNCTION
# ==========================================

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text

# ==========================================
# EMOTIONAL GUIDANCE FUNCTION
# ==========================================

def emotional_guidance(emotion):

    guidance = {

        "Anxiety":
        "Take deep breaths and try grounding exercises. Talking to someone you trust may help.",

        "Depression":
        "Remember that your feelings are valid. Consider reaching out to supportive friends or professionals.",

        "Stress":
        "Take a short break, hydrate yourself, and organize tasks one step at a time.",

        "Suicidal":
        "Please talk to someone immediately. You are not alone and support is available.",

        "Bipolar":
        "Maintain a healthy routine and monitor emotional changes carefully.",

        "Personality disorder":
        "Self-awareness and professional guidance can greatly improve emotional balance.",

        "Normal":
        "Keep maintaining a healthy routine and positive mindset."
    }

    return guidance.get(
        emotion,
        "Take care of your mental well-being."
    )

# ==========================================
# HEADER SECTION
# ==========================================

st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>
    AI-Based Mental Health Sentiment Monitoring System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style='text-align: center; color: gray;'>
    Emotion Detection using Simple Recurrent Neural Networks
    </h3>
    """,
    unsafe_allow_html=True
)

st.write("---")

# ==========================================
# ABOUT PROJECT SECTION
# ==========================================

st.header("About the Project")

st.write("""
This project uses Artificial Intelligence and Natural Language Processing (NLP)
to analyze emotional sentiment from user text inputs.

The system helps identify emotional patterns such as anxiety, depression,
stress, suicidal thoughts, and other emotional states.

Simple Recurrent Neural Networks (SimpleRNN) are used because they can process
text sequentially and remember previous words using hidden states.

Applications of Emotional AI and NLP include:

- Mental health monitoring
- Emotional well-being analysis
- AI-based counseling systems
- Human emotion detection
- Smart healthcare systems
""")

st.write("---")

# ==========================================
# USER INPUT SECTION
# ==========================================

st.header("Enter Your Thoughts or Feelings")

st.write("### Sample Sentences")

st.info("I feel very stressed because of my exams.")
st.info("Nobody understands me and I feel lonely.")
st.info("I am happy and excited about my future.")
st.info("I feel hopeless and emotionally tired.")

# Multi-line text input
user_input = st.text_area(
    "Enter your thoughts or feelings here...",
    height=200
)

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("Analyze Emotion"):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        # ==================================
        # PREPROCESS INPUT
        # ==================================

        processed_text = preprocess_text(
            user_input
        )

        # ==================================
        # TEXT TO SEQUENCE
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

        prediction = model.predict(padded)

        predicted_index = np.argmax(prediction)

        confidence = np.max(prediction)

        predicted_emotion = encoder.inverse_transform(
            [predicted_index]
        )[0]

        # ==================================
        # OUTPUT SECTION
        # ==================================

        st.write("---")

        st.header("Prediction Output")

        st.success(
            f"Emotion Detected: {predicted_emotion}"
        )

        st.info(
            f"Confidence Score: {round(confidence * 100, 2)}%"
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
                "Emotional Status: Needs Emotional Attention"
            )

        else:

            st.success(
                "Emotional Status: Emotionally Stable"
            )

        # ==================================
        # VISUALIZATION SECTION
        # ==================================

        st.write("---")

        st.header("Sentiment Confidence Visualization")

        emotions = encoder.classes_

        probabilities = prediction[0]

        fig, ax = plt.subplots(figsize=(10,5))

        ax.bar(
            emotions,
            probabilities
        )

        ax.set_xlabel("Emotions")

        ax.set_ylabel("Confidence")

        ax.set_title("Emotion Probability Distribution")

        plt.xticks(rotation=20)

        st.pyplot(fig)

        # ==================================
        # EMOTIONAL GUIDANCE SECTION
        # ==================================

        st.write("---")

        st.header("Emotional Wellness Guidance")

        guidance_message = emotional_guidance(
            predicted_emotion
        )

        st.success(guidance_message)

        st.write("""
        ### Positive Wellness Tips

        - Stay hydrated
        - Practice mindfulness
        - Talk with trusted people
        - Maintain healthy sleep
        - Exercise regularly
        - Take breaks from stress
        - Seek professional help if needed
        """)