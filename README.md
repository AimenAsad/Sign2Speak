✋ Sign2Speak

Sign2Speak is a deep learning–powered application that translates American Sign Language (ASL) alphabet signs into text and speech in real time. The app uses MediaPipe for hand landmark detection and a CNN/LSTM model (trained on ASL alphabet dataset) for classification, all wrapped inside a Streamlit interface.

🚀 Features

Real-time ASL recognition using webcam
Hand landmark extraction powered by MediaPipe
Deep learning model (CNN/LSTM) trained on ASL alphabet
Text-to-Speech (TTS) to vocalize recognized signs
Streamlit web app for easy deployment & interaction

📂 Project Structure
Sign2Speak/
│── app.py<br>                    
│── models/<br>
│   └── model_alphabet.h5 <br>    
│── processed_data/<br>
│   └── asl_alphabet_actions.npy<br> 
│── requirements.txt<br>
│── README.md<br>

⚙️ Installation

Clone the repository
git clone https://github.com/yourusername/Sign2Speak.git
cd Sign2Speak

Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

Install dependencies
pip install -r requirements.txt

⚠️ Make sure your TensorFlow & Keras versions match the one used for training (example below).
streamlit==1.32.0<br>
tensorflow==2.12.0<br>
keras==2.12.0<br>
opencv-python-headless==4.8.1.78<br>
numpy==1.26.4<br>
mediapipe==0.10.9<br>

▶️ Usage
Run the Streamlit app
streamlit run app.py

The browser will open automatically.
Grant camera access to the app.
Start signing! The recognized alphabet will appear on the screen and be spoken out loud.

🧠 Model Details

The model is a CNN + LSTM hybrid trained on ASL alphabet dataset.
Input: 21 hand landmarks (x,y,z) from MediaPipe
Output: 26 classes (A–Z)
Saved in .h5 format (models/model_alphabet.h5).


📊 Example Workflow

Capture Frame from webcam
Extract Landmarks using MediaPipe Hands
Preprocess Sequence into fixed length input
Predict Alphabet with trained CNN/LSTM
Display Output with Streamlit

📌 Roadmap / Future Improvements

Extend to full words & sentences
Support multiple signers
Add multi-language TTS support
Deploy on Streamlit Cloud / HuggingFace Spaces

🤝 Contributing

Contributions are welcome!
Fork the repo
Create a new branch (feature/new-feature)
Commit changes
Open a Pull Request
