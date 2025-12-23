# 🎮 Gesture-Controlled YouTube Player using AI

Control YouTube videos hands-free using real-time hand gestures powered by Computer Vision and Deep Learning.
This project uses a webcam to recognize hand gestures and maps them to media controls like play/pause, volume control, forwarding and rewinding.

# 🎯 Features

- Automatically opens a YouTube video

- Real-time hand tracking using MediaPipe

- Deep learning–based gesture recognition

- Hands-free media control

- Low-latency and real-time performance

- Cooldown mechanism to prevent repeated actions

#  Supported Gestures & Actions

| Gesture     | Action                  |
| ----------- | ----------------------- |
| Right       | Skip forward 10 seconds |
| Left        | Rewind 10 seconds       |
| Volume Up   | Increase volume         |
| Volume Down | Decrease volume         |
| Pause       | Play / Pause video      |

# How It Works

1)Webcam Capture

-OpenCV captures real-time video frames.

2)Hand Landmark Detection

-MediaPipe detects 21 hand landmarks per frame.

3)Feature Extraction

-Only (x, y) coordinates are extracted → 42 features.

4)Gesture Classification

-A trained TensorFlow/Keras model predicts the gesture.

5)Action Execution

-pyautogui triggers keyboard/media commands based on the gesture.

6)Live Feedback

-Predicted gesture is displayed on the webcam feed.

# Project Structure

```text
.
├── Dataa/
│   ├── Right.csv
│   ├── Left.csv
│   ├── Volume up.csv
│   ├── Volume down.csv
│   ├── pause.csv
│   └── final_dataset.csv
├── sumith_ai_model_2.keras   # Trained gesture recognition model
├── merging.py               # Dataset merging & preprocessing
├── model.ipynb              # Model training notebook
├── test.py                  # Gesture-controlled YouTube system
└── README.md
```
# How to Run

1)Install Dependencies
```bash
pip install opencv-python mediapipe tensorflow pyautogui numpy
```

2)Update Model Path
```python
model = tf.keras.models.load_model("path_to_your_model.keras")
```

3)Run the Application
```bash
Run the Application
```
4)Show Gestures in Front of Webcam

Press q to exit.

# 👤 Author

Ch. Mounika

AI / Machine Learning Enthusiast
