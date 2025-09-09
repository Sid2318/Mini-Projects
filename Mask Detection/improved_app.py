import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import time

# Set page configuration
st.set_page_config(page_title="Mask Detection", layout="wide")

# Constants
IMG_SIZE = 100
MODEL_PATH = 'mask_detector_vgg16.h5'
CONFIDENCE_THRESHOLD = 0.6  # Adjust for more accurate predictions

@st.cache_resource
def load_models():
    # Load face detector (using more reliable DNN face detector if available)
    try:
        # Try DNN face detector (better but requires opencv-contrib-python)
        face_net = cv2.dnn.readNetFromCaffe(
            'deploy.prototxt', 
            'res10_300x300_ssd_iter_140000.caffemodel'
        )
        detector_type = "dnn"
    except:
        # Fallback to Haar cascade
        face_net = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        detector_type = "haar"
    
    # Load mask detector model
    mask_net = load_model(MODEL_PATH)
    
    return face_net, detector_type, mask_net

# Debug mode toggle
debug_mode = st.sidebar.checkbox("Enable Debug Mode")

# Model parameters
confidence_threshold = st.sidebar.slider(
    "Face Detection Confidence", 0.1, 1.0, CONFIDENCE_THRESHOLD
)

# Load models
face_net, detector_type, mask_net = load_models()

if debug_mode:
    st.sidebar.write(f"Using {detector_type} face detector")
    st.sidebar.write(f"Model input shape: {mask_net.input_shape}")

def detect_faces_dnn(frame):
    """Detect faces using OpenCV DNN face detector"""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Ensure box is within frame boundaries
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)
            face = frame[startY:endY, startX:endX]
            if face.size > 0:
                faces.append(((startX, startY, endX-startX, endY-startY), face))
    return faces

def detect_faces_haar(frame):
    """Detect faces using OpenCV Haar Cascade face detector"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rect = face_net.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces = []
    for (x, y, w, h) in faces_rect:
        face = frame[y:y+h, x:x+w]
        if face.size > 0:
            faces.append(((x, y, w, h), face))
    return faces

def predict_mask(face_img):
    """Predict mask presence in a face image"""
    try:
        # Ensure face is not too small
        if face_img.shape[0] < 20 or face_img.shape[1] < 20:
            return 'Unknown', 0.5
        
        # Preprocess face image
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        
        # Make prediction
        start = time.time()
        prediction = mask_net.predict(face_img, verbose=0)
        pred_time = time.time() - start
        
        # Get label and probability
        prob = float(prediction[0][0])
        label = 'No Mask' if prob < 0.5 else 'Mask'
        
        if debug_mode:
            st.sidebar.write(f"Prediction: {label}, Probability: {prob:.4f}, Time: {pred_time:.4f}s")
            
        return label, prob
    except Exception as e:
        if debug_mode:
            st.sidebar.write(f"Prediction error: {str(e)}")
        return 'Error', 0.5

def process_frame(frame):
    """Process a single frame, detecting faces and predicting mask presence"""
    # Detect faces
    if detector_type == "dnn":
        faces = detect_faces_dnn(frame)
    else:
        faces = detect_faces_haar(frame)
    
    # Process each face
    for (face_box, face_img) in faces:
        (x, y, w, h) = face_box
        
        # Predict mask presence
        label, prob = predict_mask(face_img)
        
        # Determine color based on label
        if label == 'Mask':
            color = (0, 255, 0)  # Green for mask
        elif label == 'No Mask':
            color = (0, 0, 255)  # Red for no mask
        else:
            color = (255, 255, 0)  # Yellow for uncertain
        
        # Draw face box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Add confidence text
        conf_label = f"{label} ({prob:.2f})"
        cv2.putText(frame, conf_label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame, len(faces)

def main():
    """Main function for Streamlit UI"""
    st.title("Face Mask Detection")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Controls")
        start_camera = st.button("Start Camera")
        stop_camera = st.button("Stop Camera")
        
        if debug_mode:
            st.subheader("Debug Info")
            fps_text = st.empty()
            face_count_text = st.empty()
    
    with col1:
        frame_window = st.image([])
    
    camera_on = False
    if start_camera:
        camera_on = True
    if stop_camera:
        camera_on = False
    
    if camera_on:
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        frame_count = 0
        
        while camera_on:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
            
            # Process frame
            processed_frame, face_count = process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            
            # Show FPS on frame
            if debug_mode:
                cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                fps_text.text(f"FPS: {fps:.2f}")
                face_count_text.text(f"Faces detected: {face_count}")
            
            # Display the frame
            frame_window.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            
            # Check if stop button was pressed
            if stop_camera:
                camera_on = False
        
        cap.release()
    else:
        st.write("Click 'Start Camera' to begin face mask detection")

if __name__ == "__main__":
    main()
