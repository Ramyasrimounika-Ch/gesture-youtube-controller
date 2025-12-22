import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import tensorflow as tf
import webbrowser

# Open YouTube video in default browser
video_url = "https://www.youtube.com/watch?v=q2aENKR59w4"  # replace with your video link
webbrowser.open(video_url)
time.sleep(5)  # wait for browser/video to load

# Load your trained gesture model
model = tf.keras.models.load_model(r'D:/youtube_ai/main_model.keras')
labels = ["Right", "Left", "Volume down", "Volume up", "pause"]

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    prev_action_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract only x and y landmarks (ignore z)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y])
                landmarks = np.array(landmarks).flatten().reshape(1, -1)

                # Predict gesture
                pred = model.predict(landmarks)
                gesture = labels[np.argmax(pred)]

                # Control YouTube with 1-second cooldown
                if time.time() - prev_action_time > 1:
                    if gesture == "Right":
                        # Press 'right' twice → skip 10 seconds
                        pyautogui.press("right")
                        time.sleep(0.1)
                        pyautogui.press("right")
                    elif gesture == "Left":
                        # Press 'left' twice → go back 10 seconds
                        pyautogui.press("left")
                        time.sleep(0.1)
                        pyautogui.press("left")
                    elif gesture == "Volume up":
                        pyautogui.press("volumeup")
                    elif gesture == "Volume down":
                        pyautogui.press("volumedown")
                    elif gesture == "pause":
                        pyautogui.press("space")  # Play/Pause
                    prev_action_time = time.time()

                cv2.putText(frame, gesture, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
