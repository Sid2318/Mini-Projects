import cv2
import numpy as np
import mediapipe as mp
import os

# ---- helpers ----
def overlay_rgba(dst, src_rgb, mask_a, x, y):
    h, w = src_rgb.shape[:2]
    roi = dst[y:y+h, x:x+w]

    if mask_a.ndim == 3:
        mask_a = mask_a[..., 0]

    mask_a = mask_a.astype(np.float32)
    for c in range(3):
        roi[..., c] = (mask_a * src_rgb[..., c] + (1 - mask_a) * roi[..., c]).astype(np.uint8)

    dst[y:y+h, x:x+w] = roi


def warp_png_to_quad(frame, png_rgba, dst_quad):
    h0, w0 = png_rgba.shape[:2]
    src_quad = np.float32([[0,0],[w0,0],[w0,h0],[0,h0]])
    M = cv2.getPerspectiveTransform(src_quad, dst_quad.astype(np.float32))

    color = png_rgba[..., :3]
    alpha = png_rgba[..., 3] / 255.0

    xmin = max(int(np.floor(dst_quad[:,0].min())), 0)
    ymin = max(int(np.floor(dst_quad[:,1].min())), 0)
    xmax = min(int(np.ceil(dst_quad[:,0].max())), frame.shape[1]-1)
    ymax = min(int(np.ceil(dst_quad[:,1].max())), frame.shape[0]-1)
    if xmax <= xmin or ymax <= ymin:
        return

    out_w, out_h = xmax - xmin + 1, ymax - ymin + 1
    T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float32)
    M_shift = T @ M

    warped_color = cv2.warpPerspective(
        color, M_shift, (out_w, out_h), 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0,0,0)
    )
    warped_alpha = cv2.warpPerspective(
        alpha, M_shift, (out_w, out_h), 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=0
    )

    overlay_rgba(frame, warped_color, warped_alpha, xmin, ymin)


# ---- setup ----
# Load multiple glasses
glasses_files = ["GlassME/glasses1.png", "GlassME/glasses2.png", "GlassME/glasses3.png", "GlassME/glasses4.png", "GlassME/glasses5.png"]
glasses_list = []

for f in glasses_files:
    g = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    if g is None or g.shape[2] != 4:
        print(f"Warning: {f} not found or not RGBA. Skipping.")
    else:
        glasses_list.append(g)

if not glasses_list:
    raise FileNotFoundError("No valid glasses found!")

# ---- User selects glass at start ----
print("Available glasses:")
for idx, f in enumerate(glasses_files):
    print(f"{idx+1}. {os.path.basename(f)}")

while True:
    try:
        choice = int(input(f"Enter glass number (1-{len(glasses_list)}): "))
        if 1 <= choice <= len(glasses_list):
            current_glass = choice - 1
            break
        else:
            print("Invalid number, try again.")
    except ValueError:
        print("Please enter a number.")

# ---- FaceMesh setup ----
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                             min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Landmarks
L_OUT, L_IN, R_IN, R_OUT = 33, 133, 362, 263
L_BROW, R_BROW = 105, 334

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

with face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            fl = res.multi_face_landmarks[0].landmark
            def pt(i): return np.array([fl[i].x * w, fl[i].y * h], dtype=np.float32)

            pL_out, pL_in = pt(L_OUT), pt(L_IN)
            pR_in, pR_out = pt(R_IN), pt(R_OUT)
            pL_brow, pR_brow = pt(L_BROW), pt(R_BROW)

            left_eye_center  = 0.5 * (pL_out + pL_in)
            right_eye_center = 0.5 * (pR_in  + pR_out)
            eye_width = np.linalg.norm(right_eye_center - left_eye_center)

            width_scale = 2.1
            height_scale = 0.9
            rect_w = eye_width * width_scale
            rect_h = eye_width * height_scale

            center = 0.5 * (left_eye_center + right_eye_center)
            direction = (right_eye_center - left_eye_center)
            angle = np.arctan2(direction[1], direction[0])
            cos, sin = np.cos(angle), np.sin(angle)

            hw, hh = rect_w/2, rect_h/2
            local = np.float32([[-hw,-hh],[ hw,-hh],[ hw, hh],[-hw, hh]])
            R = np.float32([[cos,-sin],[sin,cos]])
            quad = (local @ R.T) + center

            warp_png_to_quad(frame, glasses_list[current_glass], quad)

        cv2.putText(frame, f"Glass: {current_glass+1}/{len(glasses_list)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("User Selected Glasses", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
