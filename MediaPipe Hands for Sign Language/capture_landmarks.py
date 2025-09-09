import cv2
import mediapipe as mp
import numpy as np
import os

# setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
data_dir = "sign_data"
word = "HELLO"
os.makedirs(f"{data_dir}/{word}", exist_ok=True)

sequence = []
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
    
    cv2.imshow("Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

np.save(f"{data_dir}/{word}/seq_0.npy", sequence)
cap.release()
cv2.destroyAllWindows()
