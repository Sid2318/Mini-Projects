import cv2

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the glasses image (with alpha channel, PNG)
glasses = cv2.imread("FaceDetection/glasses.png", cv2.IMREAD_UNCHANGED)
if glasses is None:
    raise FileNotFoundError("glasses.png not found. Please check path!")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Resize glasses to fit face width
        glasses_resized = cv2.resize(glasses, (w, int(glasses.shape[0] * w / glasses.shape[1])))

        # Coordinates for overlay
        y1 = y + int(h / 4)  # place glasses on upper part of face
        y2 = y1 + glasses_resized.shape[0]
        x1 = x
        x2 = x + glasses_resized.shape[1]

        # Check bounds to avoid errors
        if y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
            continue

        # Split channels
        alpha_glasses = glasses_resized[:, :, 3] / 255.0
        rgb_glasses = glasses_resized[:, :, :3]

        # Overlay with transparency
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (
                alpha_glasses * rgb_glasses[:, :, c] +
                (1 - alpha_glasses) * frame[y1:y2, x1:x2, c]
            )

    cv2.imshow("Face Detection with Glasses", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
