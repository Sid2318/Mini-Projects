import cv2
import mediapipe as mp
import os
import numpy as np
import json
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# MediaPipe models and utilities for pose, face, and hand landmark detection
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def mediapipe_detection(image, model):
    """
    Process an image with MediaPipe holistic model to detect landmarks
    
    Args:
        image: Input BGR image from OpenCV
        model: MediaPipe holistic model instance
    
    Returns:
        image: Processed BGR image
        results: MediaPipe detection results
    """
    # Convert BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Set image as read-only before processing
    image.flags.writeable = False
    # Process the image with MediaPipe
    results = model.process(image)
    # Set image as writable again
    image.flags.writeable = True
    # Convert RGB image back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """
    Extract keypoints from MediaPipe detection results
    
    Args:
        results: MediaPipe detection results
    
    Returns:
        numpy array: Flattened array of all keypoints
    """
    # Extract pose landmarks (x, y, z, visibility) if detected, otherwise zeros
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Extract face landmarks (x, y, z) if detected, otherwise zeros
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    # Extract left hand landmarks (x, y, z) if detected, otherwise zeros
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Extract right hand landmarks (x, y, z) if detected, otherwise zeros
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Concatenate all keypoints into a single array
    return np.concatenate([pose, face, lh, rh])

def main():
    """Main function to process sign language videos and extract features using MediaPipe"""
    
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VIDEO_PATH = os.path.join(BASE_DIR, "archive", "videos")
    JSON_PATH = os.path.join(BASE_DIR, "archive", "WLASL_v0.3.json")
    DATA_PATH = os.path.join(BASE_DIR, "MP_Data_Frames")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created directory: {DATA_PATH}")
    
    print("Loading metadata from JSON file...")
    # Read metadata from JSON file
    with open(JSON_PATH, 'r') as file:
        metadata = json.load(file)
    
    # Create label map: video_id -> [label, frame_start, frame_end, fps]
    print("Creating label map...")
    labelMap = {}
    for i in metadata:
        label = i['gloss']  # The sign word/label
        for instance in i['instances']:
            video_id = int(instance['video_id'])
            frame_start = instance['frame_start']
            frame_end = instance['frame_end']
            fps = instance['fps']
            labelMap[video_id] = [label, frame_start, frame_end, fps]
    
    # Track processed videos per label (limit to 5 videos per word)
    videos_processed = {}
    total_processed = 0
    
    print(f"Found {len(labelMap)} videos with metadata")
    print("Starting video processing...")
    
    # Process each video file in the dataset
    for video in os.listdir(VIDEO_PATH):
        if video.endswith('.mp4'):
            video_id = int(os.path.splitext(video)[0])
            
            # Check if this video has metadata
            if video_id in labelMap:
                label, start_frame, end_frame, fps = labelMap[video_id]
                
                # Initialize counter for this label if not already done
                if label not in videos_processed:
                    videos_processed[label] = 0
                
                # Limit to 5 videos per label
                if videos_processed[label] < 5:
                    print(f"Processing video {video} for label '{label}' ({videos_processed[label]+1}/5)")
                    
                    # Open video file
                    cap = cv2.VideoCapture(os.path.join(VIDEO_PATH, video))
                    cap.set(cv2.CAP_PROP_FPS, fps)
                    
                    # Initialize MediaPipe holistic model
                    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                        # Create directory for this label
                        action_path = os.path.join(DATA_PATH, label)
                        if not os.path.exists(action_path):
                            os.makedirs(action_path)
                        
                        # Create directory for this specific video
                        video_dir = os.path.join(action_path, str(video_id))
                        if not os.path.exists(video_dir):
                            os.makedirs(video_dir)
                        
                        # Process each frame of the video
                        frame_count = 0
                        frames_saved = 0
                        
                        while cap.isOpened():
                            success, image = cap.read()
                            if not success:
                                break
                            
                            frame_count += 1
                            
                            # Skip frames outside the specified range
                            if frame_count < start_frame or (end_frame != -1 and frame_count > end_frame):
                                continue
                            
                            # Process the frame with MediaPipe
                            image, results = mediapipe_detection(image, holistic)
                            
                            # Skip frames where no keypoints were detected
                            if not results.pose_landmarks and not results.face_landmarks and not results.left_hand_landmarks and not results.right_hand_landmarks:
                                print(f"No landmarks detected in frame {frame_count}")
                                continue
                            
                            # Extract keypoints from detection results
                            keypoints = extract_keypoints(results)
                            
                            # Save keypoints to a numpy file
                            np.save(os.path.join(video_dir, f'{frames_saved}.npy'), keypoints)
                            frames_saved += 1
                        
                        print(f"Saved {frames_saved} frames for video {video_id}")
                        cap.release()
                    
                    videos_processed[label] += 1
                    total_processed += 1
    
    print(f"Processing complete! Processed {total_processed} videos across {len(videos_processed)} labels.")
    print(f"Data saved to {DATA_PATH}")

def list_processed_data(data_path):
    """
    List all processed data directories and verify the structure
    
    Args:
        data_path: Path to the directory containing processed data
    """
    if not os.path.exists(data_path):
        print(f"Data directory {data_path} does not exist!")
        return
    
    # List all action categories (sign words)
    actions = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
    print(f"Found {len(actions)} action categories: {actions}")
    
    # Get statistics for each category
    for action in actions:
        action_path = os.path.join(data_path, action)
        videos = [name for name in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, name))]
        
        total_frames = 0
        for video in videos:
            video_path = os.path.join(action_path, video)
            frames = len([f for f in os.listdir(video_path) if f.endswith('.npy')])
            total_frames += frames
        
        print(f"  - {action}: {len(videos)} videos, {total_frames} total frames")

def prepare_training_data(data_path):
    """
    Prepare training data from processed keypoints
    
    Args:
        data_path: Path to the directory containing processed data
    
    Returns:
        X: Feature array
        y: Label array
        actions: List of action classes
    """
    # Get list of actions (sign words)
    actions = np.array([name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))])
    
    # Create a mapping from label to numeric index
    label_map = {label: num for num, label in enumerate(actions)}
    print(f"Label map: {label_map}")
    
    sequences, labels = [], []
    
    # Process each action category
    for action in actions:
        # Get all video sequences for this action
        for sequence in os.listdir(os.path.join(data_path, action)):
            sequence_path = os.path.join(data_path, action, sequence)
            
            if os.path.isdir(sequence_path):
                window = []
                # Get all frames for this video
                frames = [f for f in os.listdir(sequence_path) if f.endswith('.npy')]
                frames = sorted(frames, key=lambda x: int(os.path.splitext(x)[0]))
                
                # Load each frame's keypoints
                for frame in frames:
                    res = np.load(os.path.join(sequence_path, frame))
                    window.append(res)
                
                # Add this sequence and its label
                sequences.append(window)
                labels.append(label_map[action])
    
    # Return as numpy arrays
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"Data shape: {X.shape} - {len(actions)} classes, {len(sequences)} sequences")
    
    return X, y, actions

def train_model(X, y, actions):
    """
    Train LSTM model for sign language recognition
    
    Args:
        X: Feature array
        y: Label array
        actions: List of action classes
    
    Returns:
        Trained model
    """
    
    # Convert labels to one-hot encoding
    y = to_categorical(y).astype(int)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Create logs directory for TensorBoard
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    tb_callback = TensorBoard(log_dir=log_dir)
    
    # Define LSTM model architecture
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    
    # Compile and train the model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    print("Training model...")
    model.fit(X_train, y_train, epochs=50, callbacks=[tb_callback])
    
    # Print model summary
    model.summary()
    
    return model

if __name__ == "__main__":
    # Step 1: Process videos and extract features
    print("Step 1: Processing videos and extracting features...")
    main()
    
    # Step 2: List processed data
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MP_Data_Frames")
    print("\nStep 2: Listing processed data...")
    list_processed_data(DATA_PATH)
    
    # Step 3: Prepare training data
    print("\nStep 3: Preparing training data...")
    try:
        X, y, actions = prepare_training_data(DATA_PATH)
        
        # Step 4: Train the model
        print("\nStep 4: Training model...")
        model = train_model(X, y, actions)
        print("Training complete!")
    except Exception as e:
        print(f"Error during training: {e}")
        
