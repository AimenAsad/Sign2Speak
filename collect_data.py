import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Setup MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Data Collection Parameters ---
DATA_PATH = os.path.join('MP_Data')

# ✅ Define ASL Alphabets (A–Z)
actions = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])

no_sequences = 30       # number of repetitions per alphabet
sequence_length = 30    # number of frames per repetition

# --- Create Folders ---
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except FileExistsError:
            pass

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
else:
    print("Webcam opened successfully. Starting alphabet data collection...")
    print("Press 'q' to quit at any time.")
    time.sleep(1)

    for action in actions:
        print(f"\n--- Collecting data for alphabet: {action} ---")
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: Failed to grab frame.")
                    break

                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_bgr,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                        )

                # --- Show Status ---
                if frame_num == 0:
                    cv2.putText(frame_bgr, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame_bgr, f'Collecting "{action}" | Video: {sequence + 1}/{no_sequences}',
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Sign2Speak Data Collection', frame_bgr)
                    cv2.waitKey(2000)  # pause before each sequence
                else:
                    cv2.putText(frame_bgr, f'Collecting "{action}" | Video: {sequence + 1}/{no_sequences}',
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Sign2Speak Data Collection', frame_bgr)

                # --- Extract Keypoints ---
                keypoints = []
                if results.multi_hand_landmarks:
                    for landmark in results.multi_hand_landmarks[0].landmark:
                        keypoints.extend([landmark.x, landmark.y, landmark.z])
                else:
                    keypoints = list(np.zeros(21 * 3))

                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting early.")
                    cap.release()
                    hands.close()
                    cv2.destroyAllWindows()
                    exit()

    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("Alphabet data collection completed successfully.")
