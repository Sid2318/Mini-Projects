import cv2
import numpy as np
import mediapipe as mp

# ---- helpers ----
def overlay_rgba(dst, src_rgb, mask_a, x, y):
    """
    Alpha-blend src_rgb (HxWx3) onto dst (HxWx3) at top-left (x,y) using mask_a (HxW, 0..1).
    """
    h, w = src_rgb.shape[:2]
    roi = dst[y:y+h, x:x+w]

    # Ensure mask_a is 2D
    if mask_a.ndim == 3:
        mask_a = mask_a[..., 0]

    mask_a = mask_a.astype(np.float32)

    for c in range(3):
        roi[..., c] = (mask_a * src_rgb[..., c] + (1 - mask_a) * roi[..., c]).astype(np.uint8)

    dst[y:y+h, x:x+w] = roi


def warp_png_to_quad(frame, png_rgba, dst_quad):
    """
    Warp png_rgba (HxWx4) to dst_quad (4x2 float32, order: tl,tr,br,bl) on frame with alpha.
    """
    h0, w0 = png_rgba.shape[:2]
    src_quad = np.float32([[0,0],[w0,0],[w0,h0],[0,h0]])
    M = cv2.getPerspectiveTransform(src_quad, dst_quad.astype(np.float32))

    # Split channels
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

    warped_color = cv2.warpPerspective(color, M_shift, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    warped_alpha = cv2.warpPerspective(alpha, M_shift, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    overlay_rgba(frame, warped_color, warped_alpha, xmin, ymin)


# ---- setup ----
glasses = cv2.imread("FaceDetection/glasses.png", cv2.IMREAD_UNCHANGED)
if glasses is None or glasses.shape[2] != 4:
    raise FileNotFoundError("glasses.png not found or missing alpha channel. Place a RGBA file next to this script.")

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# Landmarks we’ll use (MediaPipe FaceMesh indices)
# Left eye outer corner: 33, inner: 133
# Right eye inner: 362, outer: 263
# Eyebrow top-ish points for vertical placement (e.g., 105 left brow, 334 right brow)
L_OUT, L_IN, R_IN, R_OUT = 33, 133, 362, 263
L_BROW, R_BROW = 105, 334

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# For camera intrinsics approximation (only used if needed later)
with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            fl = res.multi_face_landmarks[0].landmark
            # Get key points (pixel coords)
            def pt(i):
                return np.array([fl[i].x * w, fl[i].y * h], dtype=np.float32)

            pL_out, pL_in = pt(L_OUT), pt(L_IN)
            pR_in,  pR_out = pt(R_IN),  pt(R_OUT)
            pL_brow, pR_brow = pt(L_BROW), pt(R_BROW)

            # Eye line: we form a quad slightly above/below the eye line to place glasses
            left_eye_center  = 0.5 * (pL_out + pL_in)
            right_eye_center = 0.5 * (pR_in  + pR_out)
            eye_width = np.linalg.norm(right_eye_center - left_eye_center)

            # Define a rectangle around the eyes in destination space
            # Expand a bit beyond eyes for a nicer fit
            width_scale = 2.1      # widen beyond eye distance
            height_scale = 0.9     # glasses height relative to width
            rect_w = eye_width * width_scale
            rect_h = eye_width * height_scale

            # Build an oriented quad centered mid-eye, aligned to the eye direction
            center = 0.5 * (left_eye_center + right_eye_center)
            direction = (right_eye_center - left_eye_center)
            angle = np.arctan2(direction[1], direction[0])
            cos, sin = np.cos(angle), np.sin(angle)

            # Rectangle corners in local space (tl,tr,br,bl)
            hw, hh = rect_w/2, rect_h/2
            local = np.float32([[-hw,-hh],[ hw,-hh],[ hw, hh],[-hw, hh]])
            R = np.float32([[cos,-sin],[sin,cos]])
            quad = (local @ R.T) + center  # (4,2)

            # Warp the PNG onto this quad
            warp_png_to_quad(frame, glasses, quad)

        cv2.imshow("Pseudo-3D Glasses (FaceMesh)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
