✋ Sign2Speak

Sign2Speak is a deep learning–powered application that translates American Sign Language (ASL) alphabet signs into text and speech in real time. The app uses MediaPipe for hand landmark detection and a CNN/LSTM model (trained on ASL alphabet dataset) for classification, all wrapped inside a Streamlit interface.

🚀 Features

Real-time ASL recognition using webcam<br>
Hand landmark extraction powered by MediaPipe<br>
Deep learning model (CNN/LSTM) trained on ASL alphabet<br>
Text-to-Speech (TTS) to vocalize recognized signs<br>
Streamlit web app for easy deployment & interaction

📂 Project Structure
Sign2Speak/<br>
│── app.py<br>                    
│── models/<br>
│   └── model_alphabet.h5 <br>    
│── processed_data/<br>
│   └── asl_alphabet_actions.npy<br> 
│── requirements.txt<br>
│── README.md<br>

🧠 Model Details

The model is a CNN + LSTM hybrid trained on ASL alphabet dataset.<br>
Input: 21 hand landmarks (x,y,z) from MediaPipe.<br>
Output: 26 classes (A–Z)<br>
Saved in .h5 format (models/model_alphabet.h5).


📊 Example Workflow

Capture Frame from webcam<br>
Extract Landmarks using MediaPipe Hands<br>
Preprocess Sequence into fixed length input<br>
Predict Alphabet with trained CNN/LSTM<br>
Display Output with Streamlit<br>

📌 Roadmap / Future Improvements

Extend to full words & sentences<br>
Support multiple signers<br>
Add multi-language TTS support<br>
Deploy on Streamlit Cloud / HuggingFace Spaces

🤝 Contributing

Contributions are welcome!<br>
Fork the repo<br>
Create a new branch (feature/new-feature)<br>
Commit changes<br>
Open a Pull Request
