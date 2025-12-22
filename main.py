import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model(r"D:/sumith_ai/main_model.keras")


# Define label list as per your model training
labels = [
    "Right",
    "Left",    
    "Volume down",            
    "Volume up",          
    "pause"            
]

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # Mirror the image for natural interaction
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 (x, y) landmarks -> 42 features
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y])  # normalized values (0 to 1)

            if len(features) == 42:
                # Reshape and predict
                input_array = np.array(features).reshape(1, -1)
                prediction = model.predict(input_array)
                predicted_index = np.argmax(prediction)
                confidence = prediction[0][predicted_index]
                predicted_label = labels[predicted_index]

                # Display prediction on frame
                cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                            2, (0, 255, 0), 3)

    cv2.imshow("ASL Hand Sign Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()