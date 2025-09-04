import cv2
import numpy as np
import os
import logging
from flask import Flask, Response, request
import mediapipe as mp

# Disable logs
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.disable(logging.WARNING)

# Flask app
app = Flask(__name__)

# ---- helpers (same as before) ----
def overlay_rgba(dst, src_rgb, mask_a, x, y):
    h, w = src_rgb.shape[:2]
    roi = dst[y:y+h, x:x+w]

    if mask_a.ndim == 3:
        mask_a = mask_a[..., 0]

    # mask_a = (mask_a / 255.0).astype(np.float32)
    mask_a = mask_a.astype(np.float32)
    for c in range(3):
        roi[..., c] = (mask_a * src_rgb[..., c] + (1 - mask_a) * roi[..., c]).astype(np.uint8)

    dst[y:y+h, x:x+w] = roi
    # print(f"[DEBUG] Overlay applied at x:{x}, y:{y}, w:{w}, h:{h}")

def warp_png_to_quad(frame, png_rgba, dst_quad):
    h0, w0 = png_rgba.shape[:2]
    # print(f"[DEBUG] Warping PNG of shape: {png_rgba.shape}")
    
    src_quad = np.float32([[0,0],[w0,0],[w0,h0],[0,h0]])
    M = cv2.getPerspectiveTransform(src_quad, dst_quad.astype(np.float32))

    color = png_rgba[..., :3]
    alpha = png_rgba[..., 3] / 255.0

    xmin = max(int(np.floor(dst_quad[:,0].min())), 0)
    ymin = max(int(np.floor(dst_quad[:,1].min())), 0)
    xmax = min(int(np.ceil(dst_quad[:,0].max())), frame.shape[1]-1)
    ymax = min(int(np.ceil(dst_quad[:,1].max())), frame.shape[0]-1)
    if xmax <= xmin or ymax <= ymin:
        print(f"[DEBUG] Invalid quad coordinates: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
        return

    out_w, out_h = xmax - xmin + 1, ymax - ymin + 1
    T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float32)
    M_shift = T @ M

    warped_color = cv2.warpPerspective(color, M_shift, (out_w, out_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    warped_alpha = cv2.warpPerspective(alpha, M_shift, (out_w, out_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    overlay_rgba(frame, warped_color, warped_alpha, xmin, ymin)
    # print(f"[DEBUG] Glass warped and overlaid.")

# ---- Glasses load ----
glasses_files = [
    r"C:\Users\siddhi\Desktop\mini\GlassME\backend\images\glasses1.png",
    r"C:\Users\siddhi\Desktop\mini\GlassME\backend\images\glasses2.png",
    r"C:\Users\siddhi\Desktop\mini\GlassME\backend\images\glasses3.png",
    r"C:\Users\siddhi\Desktop\mini\GlassME\backend\images\glasses4.png",
    r"C:\Users\siddhi\Desktop\mini\GlassME\backend\images\glasses5.png"
]

# Load glasses and verify they exist
glasses_list = []
for f in glasses_files:
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    if img is not None and img.shape[2] == 4:  # Check for RGBA image
        glasses_list.append(img)
        print(f"[DEBUG] Loaded {f} with shape {img.shape}")
    else:
        print(f"Warning: Could not load {f} or not RGBA format")


if not glasses_list:
    print("ERROR: No valid glasses images found. Check the image paths and formats.")
    # Default to a simple placeholder if no glasses are found
    glasses_list = [np.ones((100, 200, 4), dtype=np.uint8) * 255]  # White rectangle as fallback

current_glass = 0  # default
print(f"[INFO] Current glass index: {current_glass}")

# ---- Mediapipe ----
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                             min_detection_confidence=0.5, min_tracking_confidence=0.5)

L_OUT, L_IN, R_IN, R_OUT = 33, 133, 362, 263
L_BROW, R_BROW = 105, 334

# ---- Update selected glasses ----
@app.route("/set-glass", methods=["POST"])
def set_glass():
    global current_glass
    number = request.json.get("number", 1)
    print(f"[INFO] Set glass request received: {number}")
    if 1 <= number <= len(glasses_list):
        current_glass = number - 1
        print(f"[INFO] Current glass updated to index: {current_glass}")
        return {"success": True, "glass": current_glass+1}
    return {"success": False, "error": "Invalid glass number"}

# ---- Video stream ----
def generate_frames():
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

                left_eye_center  = 0.5 * (pL_out + pL_in)
                right_eye_center = 0.5 * (pR_in  + pR_out)
                eye_width = np.linalg.norm(right_eye_center - left_eye_center)

                rect_w = eye_width * 2.1
                rect_h = eye_width * 0.9
                center = 0.5 * (left_eye_center + right_eye_center)
                direction = (right_eye_center - left_eye_center)
                angle = np.arctan2(direction[1], direction[0])
                cos, sin = np.cos(angle), np.sin(angle)

                hw, hh = rect_w/2, rect_h/2
                local = np.float32([[-hw,-hh],[ hw,-hh],[ hw, hh],[-hw, hh]])
                R = np.float32([[cos,-sin],[sin,cos]])
                quad = (local @ R.T) + center

                # print(f"[DEBUG] Glass quad coordinates: {quad}")
                warp_png_to_quad(frame, glasses_list[current_glass], quad)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("[INFO] Starting Flask server on port 7000")
    app.run(host="0.0.0.0", port=7000)
