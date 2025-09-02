import cv2
import numpy as np
import mediapipe as mp

# ---- Helpers ----
def draw_3d_box(frame, rotation_vector, translation_vector, camera_matrix, dist_coeffs, size=0.06):
    """
    Draw a 3D cube as glasses.
    """
    # Define 8 corners of the cube in 3D space (centered at origin)
    half = size / 2
    cube_points_3d = np.array([
        [-half, -half, 0],
        [ half, -half, 0],
        [ half,  half, 0],
        [-half,  half, 0],
        [-half, -half, -size],
        [ half, -half, -size],
        [ half,  half, -size],
        [-half,  half, -size]
    ])

    # Project 3D points to 2D
    points_2d, _ = cv2.projectPoints(cube_points_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    points_2d = points_2d.reshape(-1,2).astype(int)

    # Draw cube edges
    # Bottom square
    cv2.polylines(frame, [points_2d[:4]], isClosed=True, color=(0,255,0), thickness=2)
    # Top square
    cv2.polylines(frame, [points_2d[4:]], isClosed=True, color=(0,255,0), thickness=2)
    # Vertical lines
    for i in range(4):
        cv2.line(frame, tuple(points_2d[i]), tuple(points_2d[i+4]), (0,255,0), 2)

# ---- MediaPipe Setup ----
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                             min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w = frame.shape[:2]

# Camera intrinsics approximation
focal_length = w
center = (w/2, h/2)
camera_matrix = np.array([[focal_length, 0, center[0]],
                          [0, focal_length, center[1]],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(4)  # assume no lens distortion

# 3D model points for head-pose (nose tip, chin, eyes, mouth corners)
model_points = np.array([
    [0.0, 0.0, 0.0],             # Nose tip
    [0.0, -0.06, -0.03],         # Chin
    [-0.03, 0.03, -0.03],        # Left eye corner
    [0.03, 0.03, -0.03],         # Right eye corner
    [-0.025, -0.03, -0.03],      # Left mouth corner
    [0.025, -0.03, -0.03]        # Right mouth corner
])

# Landmarks corresponding to model_points (MediaPipe indices)
LM_NP, LM_CH, LM_LE, LM_RE, LM_LM, LM_RM = 1, 152, 33, 263, 61, 291

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # 2D image points
        image_points = np.array([
            [lm[LM_NP].x * w, lm[LM_NP].y * h],
            [lm[LM_CH].x * w, lm[LM_CH].y * h],
            [lm[LM_LE].x * w, lm[LM_LE].y * h],
            [lm[LM_RE].x * w, lm[LM_RE].y * h],
            [lm[LM_LM].x * w, lm[LM_LM].y * h],
            [lm[LM_RM].x * w, lm[LM_RM].y * h]
        ], dtype=np.float32)

        # Solve PnP for head pose
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points,
                                                                    camera_matrix, dist_coeffs,
                                                                    flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            # Draw 3D cube (glasses)
            draw_3d_box(frame, rotation_vector, translation_vector, camera_matrix, dist_coeffs, size=0.06)

    cv2.imshow("3D Glasses Overlay", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
