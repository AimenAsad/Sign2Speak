import streamlit as st
import numpy as np
import cv2
import av
import mediapipe as mp
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Sign2Speak", page_icon="✋", layout="centered")
MODEL_PATH = "models/model_alphabet.h5"
ACTIONS_PATH = "processed_data/asl_alphabet_actions.npy"

# This should be a global variable or handled by a singleton pattern
# to avoid re-initializing on every frame.
# This is a key part of the real-time performance.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing = mp.solutions.drawing_utils

# Use st.cache_resource for heavy objects like models and Mediapipe
@st.cache_resource
def load_model_cached(path=MODEL_PATH):
    from tensorflow.keras.models import load_model
    model = load_model(path)
    return model

model = load_model_cached()
actions = np.load(ACTIONS_PATH)

# Helper: extract keypoints from mediapipe results (single hand)
def extract_keypoints_from_results(results):
    if results and results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        kp = []
        for point in lm:
            kp.extend([point.x, point.y, point.z])
        return np.array(kp, dtype=np.float32)
    else:
        return np.zeros(21 * 3, dtype=np.float32)

# --- Video Processing Class ---
# All of the frame-by-frame logic goes here
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = load_model_cached()
        self.hands = hands
        self.drawing = drawing
        self.mp_hands = mp_hands
        self.actions = actions
        
        # A queue to send data back to the main thread
        # This is how we get the prediction out of the transformer
        self.result_queue = []
    
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="rgb24")
        
        # Process the image with MediaPipe
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        
        # Draw landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Get keypoints and make a prediction
        keypoints = extract_keypoints_from_results(results)
        seq = keypoints.reshape(1, 1, -1)
        
        pred = self.model.predict(seq, verbose=0)[0]
        pred_index = int(np.argmax(pred))
        pred_label = self.actions[pred_index].upper()
        confidence = float(np.max(pred))
        
        # Put the results into the queue to be read by the main script
        # This is the thread-safe way to communicate
        if len(self.result_queue) < 10:  # Prevent the queue from growing indefinitely
            self.result_queue.append({"label": pred_label, "confidence": confidence})

        # Add prediction text to the video frame
        display_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.putText(display_bgr, f"Prediction: {pred_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_bgr, f"Confidence: {confidence:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        
        return av.VideoFrame.from_ndarray(display_rgb, format="rgb24")


# --- Streamlit UI ---
st.markdown("<h1 style='text-align:center;color:#2E86C1'>✋ Sign2Speak - ASL Alphabet Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555'>Show an ASL alphabet sign to the camera and see the prediction in real-time</p>", unsafe_allow_html=True)
st.write("---")

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_ctx = webrtc_streamer(
        key="sign-recognition",
        video_transformer_factory=VideoProcessor,
        async_processing=True
    )

with col2:
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = {"label": "N/A", "confidence": 0.0}

    # Use a loop to continuously check for new results from the processor
    if webrtc_ctx.video_transformer:
        # A simple loop to read from the queue
        while webrtc_ctx.video_transformer.result_queue:
            result = webrtc_ctx.video_transformer.result_queue.pop(0)
            st.session_state.latest_result = result
            
    # Display the results from the session state
    st.markdown(f"### Prediction: **{st.session_state.latest_result['label']}**")
    st.markdown(f"### Confidence: **{st.session_state.latest_result['confidence']:.2f}**")
