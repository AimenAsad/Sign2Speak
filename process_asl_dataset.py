import cv2
import mediapipe as mp
import numpy as np
import os

# --- Configuration ---
DATASET_ROOT = 'data/asl_alphabet_train' 
ACTIONS = np.array([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'space', 'nothing' 
])

# Where to save the processed keypoint data
OUTPUT_X_FILE = 'processed_data/X_asl_alphabet.npy'
OUTPUT_Y_FILE = 'processed_data/y_asl_alphabet.npy'

# --- MediaPipe Hands Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,     
    min_detection_confidence=0.5
)

# --- Data Storage ---
all_keypoints = [] 
all_labels = []    

# --- Process Dataset ---
print(f"Starting to process images from: {DATASET_ROOT}")

for action_idx, action_name in enumerate(ACTIONS):
    action_folder_path = os.path.join(DATASET_ROOT, action_name)

    if not os.path.exists(action_folder_path):
        print(f"Warning: Folder for action '{action_name}' not found at '{action_folder_path}'. Skipping this action.")
        continue

    print(f"  Processing action: '{action_name}' (Label: {action_idx})")
    
    # Iterate through all image files in the action's folder
    for image_file in os.listdir(action_folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(action_folder_path, image_file)
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"    Error: Could not read image '{image_path}'. Skipping.")
                continue

            # Convert BGR image to RGB for MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process the image with MediaPipe Hands
            results = hands.process(img_rgb)

            current_image_keypoints = []
            if results.multi_hand_landmarks:
                # Assuming max_num_hands=1, take the first detected hand
                for landmark in results.multi_hand_landmarks[0].landmark:
                    current_image_keypoints.extend([landmark.x, landmark.y, landmark.z])
            else:
                current_image_keypoints = list(np.zeros(21 * 3)) # 21 landmarks * 3 (x,y,z) coordinates

            all_keypoints.append(current_image_keypoints)
            all_labels.append(action_idx) 

# Release MediaPipe resources
hands.close()

# Convert lists to NumPy arrays
X_data = np.array(all_keypoints)
y_data = np.array(all_labels)

# --- Save Processed Data ---
np.save(OUTPUT_X_FILE, X_data)
np.save(OUTPUT_Y_FILE, y_data)

print(f"\n--- Data Processing Complete ---")
print(f"Total images processed: {len(all_keypoints)}")
print(f"Keypoint data shape (X): {X_data.shape}")
print(f"Label data shape (y): {y_data.shape}")
print(f"Processed data saved to '{OUTPUT_X_FILE}' and '{OUTPUT_Y_FILE}'.")