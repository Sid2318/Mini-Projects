# --------------------------- Suppress logs ---------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# --------------------------- Imports ---------------------------
import cv2
import mediapipe as mp

# --------------------------- Initialize MediaPipe Hands ---------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,        # Continuous video feed
    max_num_hands=2,                # Detect up to 2 hands
    min_detection_confidence=0.7,   # Minimum confidence to detect hand
    min_tracking_confidence=0.7     # Minimum confidence to track landmarks
)

# --------------------------- Start Webcam ---------------------------
cap = cv2.VideoCapture(0)  # Use external webcam (1)

print("Starting hand tracking. Press ESC to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip horizontally for mirror view & convert to RGB
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)

    # Draw landmarks and print coordinates
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            # Print landmark coordinates
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f"Hand {idx+1}, Landmark {id}: ({cx}, {cy})")

    # Show the frame
    cv2.imshow('MediaPipe Hands', frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
