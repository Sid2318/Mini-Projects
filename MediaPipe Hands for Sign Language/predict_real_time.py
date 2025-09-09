import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/sign_lstm.h5")
label_map = {0:"HELLO", 1:"THANKYOU"}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
sequence = []
sequence_length = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        landmarks = [coord for lm in handLms.landmark for coord in (lm.x, lm.y, lm.z)]
        sequence.append(landmarks)
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    
    if len(sequence) == sequence_length:
        pred = model.predict(np.expand_dims(sequence, axis=0))
        word = label_map[np.argmax(pred)]
        print("Predicted Word:", word)
        sequence = []

    cv2.imshow("Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
