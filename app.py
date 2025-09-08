import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# --- Load Trained Model ---
MODEL_PATH = "models/model_alphabet.h5"
model = load_model(MODEL_PATH)

# --- Load Actions (Alphabet Classes) ---
actions = np.load("processed_data/asl_alphabet_actions.npy")

# --- Mediapipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Streamlit UI ---
st.title("✋ Sign2Speak - ASL Alphabet Recognition")
st.markdown("Show an ASL alphabet sign to the camera and see the prediction in real-time.")

# Session state for camera control
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

# Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Camera"):
        st.session_state.run_camera = True
with col2:
    if st.button("⏹ Stop Camera"):
        st.session_state.run_camera = False

frame_placeholder = st.empty()
pred_placeholder = st.empty()

# --- Run Camera Loop ---
if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)

    while st.session_state.run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
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

        # Show prediction
        cv2.putText(frame, f"{pred_class} ({confidence:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Streamlit display
        frame_placeholder.image(frame, channels="BGR")
        pred_placeholder.write(f"### Prediction: **{pred_class}** (confidence: {confidence:.2f})")

    cap.release()
