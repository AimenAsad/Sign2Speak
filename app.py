import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# --- Load Trained Model ---
MODEL_PATH = "models/model_alphabet.h5"
@st.cache_resource 
def load_sign2speak_model():
    return load_model(MODEL_PATH)
model = load_sign2speak_model()

# --- Load Actions (Alphabet Classes) ---
actions = np.load("processed_data/asl_alphabet_actions.npy")

# --- Mediapipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# --- Streamlit Page Config ---
st.set_page_config(page_title="Sign2Speak", page_icon="✋", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 36px !important;
            font-weight: bold;
            color: #2E86C1;
        }
        .subtitle {
            text-align: center;
            font-size: 18px !important;
            color: #555;
            margin-bottom: 20px;
        }
        .prediction-box {
            padding: 15px;
            border-radius: 12px;
            background: #f0f2f6;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #2C3E50;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
            margin-top: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title & Subtitle ---
st.markdown('<p class="title">✋ Sign2Speak - ASL Alphabet Recognition</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Show an ASL alphabet sign to the camera and see the prediction in real-time</p>', unsafe_allow_html=True)

# --- Camera Controls ---
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

start_col, stop_col = st.columns([1, 1])
with start_col:
    if st.button("▶️ Start Camera", use_container_width=True):
        st.session_state.run_camera = True
with stop_col:
    if st.button("⏹ Stop Camera", use_container_width=True):
        st.session_state.run_camera = False

# --- Placeholders ---
frame_placeholder = st.empty()
pred_placeholder = st.empty()

# --- Run Camera Loop ---
if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)

    while st.session_state.run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Failed to access webcam.")
            break

        # Flip and convert
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks
        results = hands.process(img_rgb)

        keypoints = np.zeros(21*3)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]).flatten()

        # Predict
        prediction = model.predict(np.expand_dims([keypoints], axis=0))[0]
        pred_class = actions[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Display in Streamlit
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
        pred_placeholder.markdown(
            f'<div class="prediction-box">Prediction: <span style="color:#27AE60">{pred_class}</span><br>Confidence: {confidence:.2f}</div>',
            unsafe_allow_html=True
        )

    cap.release()
