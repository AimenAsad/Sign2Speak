# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import mediapipe as mp

st.set_page_config(page_title="Sign2Speak", page_icon="✋", layout="centered")
MODEL_PATH = "models/model_alphabet.h5"
ACTIONS_PATH = "processed_data/asl_alphabet_actions.npy"

@st.cache_resource
def load_model_cached(path=MODEL_PATH):
    model = load_model(path)
    return model


@st.cache_resource
def get_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    drawing = mp.solutions.drawing_utils
    return hands, mp_hands, drawing

model = load_model_cached()
actions = np.load(ACTIONS_PATH)
hands, mp_hands_mod, drawing = get_mediapipe()

# UI
st.markdown("<h1 style='text-align:center;color:#2E86C1'>✋ Sign2Speak - ASL Alphabet Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555'>Show an ASL alphabet sign to the camera and see the prediction in real-time</p>", unsafe_allow_html=True)
st.write("---")

# Start/Stop via session_state
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

cols = st.columns([1,1])
with cols[0]:
    if st.button("▶️ Start Camera", use_container_width=True):
        st.session_state.run_camera = True
with cols[1]:
    if st.button("⏹ Stop Camera", use_container_width=True):
        st.session_state.run_camera = False

st.info("Allow camera access when the browser prompts you. Use the camera widget below to take snapshots (or use the shutter).")

camera_col, output_col = st.columns([2,1])
with camera_col:
    uploaded_file = st.camera_input("Camera")

with output_col:
    pred_card = st.empty()
    conf_card = st.empty()

# Helper: extract keypoints from mediapipe results (single hand)
def extract_keypoints_from_results(results):
    if results and results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        kp = []
        for point in lm:
            kp.extend([point.x, point.y, point.z])
        return np.array(kp, dtype=np.float32)
    else:
        return np.zeros(21*3, dtype=np.float32)

# Process snapshot when camera is on and file provided
if st.session_state.run_camera and uploaded_file is not None:
    # Read image bytes into numpy array (RGB)
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    # Feed to MediaPipe (which expects RGB)
    results = hands.process(img_rgb)

    # Draw landmarks on a copy for display
    display_img = img_rgb.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing.draw_landmarks(display_img, hand_landmarks, mp_hands_mod.HAND_CONNECTIONS)

    # Extract keypoints, reshape for LSTM (1, 1, features)
    keypoints = extract_keypoints_from_results(results)
    seq = keypoints.reshape(1, 1, -1)  # shape (1, 1, 63)

    # Predict
    pred = model.predict(seq, verbose=0)[0]
    pred_index = int(np.argmax(pred))
    pred_label = actions[pred_index].upper()
    confidence = float(np.max(pred))

    # Overlay prediction text
    display_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
    cv2.putText(display_bgr, f"{pred_label} ({confidence:.2f})", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)

    # Show image and prediction
    st.image(display_rgb, use_container_width=True)
    pred_card.markdown(f"### Prediction: **{pred_label}**")
    conf_card.markdown(f"### Confidence: **{confidence:.2f}**")
elif not st.session_state.run_camera:
    st.info("Camera is stopped. Click ▶️ Start Camera to begin.")
else:
    st.info("No image yet — point your camera and take a snapshot using the camera widget above.")
