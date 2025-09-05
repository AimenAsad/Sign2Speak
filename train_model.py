import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import time

TRAIN_MODE = "alphabet" 
X_PATH = 'processed_data/X_asl_alphabet.npy'
y_PATH = 'processed_data/y_asl_alphabet.npy'
ACTIONS_PATH = 'processed_data/asl_alphabet_actions.npy'
MODEL_SAVE_PATH = 'models/model_alphabet.h5'

SEQUENCE_LENGTH = 1  
INPUT_FEATURES = 21 * 3  # 21 landmarks * 3 coords (x,y,z) = 63
os.makedirs('models', exist_ok=True)

# --- Data Loading ---
try:
    X = np.load(X_PATH)
    y = np.load(y_PATH)
    actions = np.load(ACTIONS_PATH)
    print(f"Loaded ASL Alphabet data:")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Number of classes (alphabets): {actions.shape[0]}")
except FileNotFoundError as e:
    print(f"Error: Missing alphabet dataset file: {e}")
    print("Make sure you ran your preprocessing script for ASL Alphabet first.")
    exit()

# --- Reshape X for LSTM input ---
if X.ndim == 2:  # (samples, features) → (samples, timesteps, features)
    X = X.reshape(X.shape[0], SEQUENCE_LENGTH, X.shape[1])

# Convert labels to categorical (one-hot)
y = to_categorical(y, num_classes=actions.shape[0]).astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
print(f"Training data shapes: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Testing data shapes: X_test {X_test.shape}, y_test {y_test.shape}")

# --- Model Definition ---
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, INPUT_FEATURES)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))  # output = 26 classes (A–Z)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- TensorBoard callback ---
log_dir = os.path.join('Logs', "alphabet_" + str(int(time.time())))
os.makedirs(log_dir, exist_ok=True)
tb_callback = TensorBoard(log_dir=log_dir)

# --- Training ---
print("\n--- Starting Model Training for ASL Alphabet ---")
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback], validation_data=(X_test, y_test))
print("--- Training Complete ---")

# --- Save Model ---
model.save(MODEL_SAVE_PATH)
print(f"Alphabet model saved as '{MODEL_SAVE_PATH}'")
