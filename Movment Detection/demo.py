"""
This script demonstrates how to use the sign language recognition system.
It provides examples of how to process videos, visualize landmarks, and train models.
"""

import os
import argparse

def check_file_exists(file_path):
    """Check if a file exists and print a message if not."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return False
    return True

def process_single_video(video_id=None):
    """Process a single video from the dataset."""
    import cv2
    import mediapipe as mp
    import numpy as np
    import json
    import os
    from main import mediapipe_detection, extract_keypoints
    
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VIDEO_PATH = os.path.join(BASE_DIR, "archive", "videos")
    JSON_PATH = os.path.join(BASE_DIR, "archive", "WLASL_v0.3.json")
    
    # Make sure the JSON file exists
    if not check_file_exists(JSON_PATH):
        return
    
    # Get list of available videos
    available_videos = [f for f in os.listdir(VIDEO_PATH) if f.endswith('.mp4')]
    
    # If no video_id is provided, use the first available video
    if video_id is None:
        if not available_videos:
            print("No videos found in the archive/videos directory.")
            return
        video_id = os.path.splitext(available_videos[0])[0]
    
    video_file = f"{video_id}.mp4"
    video_path = os.path.join(VIDEO_PATH, video_file)
    
    # Make sure the video file exists
    if not check_file_exists(video_path):
        print(f"Available videos: {available_videos[:5]}")
        return
    
    # Read metadata to get the label for this video
    with open(JSON_PATH, 'r') as file:
        metadata = json.load(file)
    
    # Find the label for this video
    label = None
    frame_start = 1
    frame_end = -1
    
    for item in metadata:
        for instance in item['instances']:
            if str(instance['video_id']) == video_id:
                label = item['gloss']
                frame_start = instance.get('frame_start', 1)
                frame_end = instance.get('frame_end', -1)
                break
        if label:
            break
    
    if not label:
        print(f"No metadata found for video {video_id}.")
        return
    
    print(f"Processing video {video_id} (Label: {label})")
    
    # Create output directory
    output_dir = os.path.join(BASE_DIR, "demo_output", label, video_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the video
    cap = cv2.VideoCapture(video_path)
    
    # Initialize MediaPipe holistic model
    mp_holistic = mp.solutions.holistic
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame_count = 0
        saved_frames = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # Skip frames outside the specified range
            if frame_count < frame_start or (frame_end != -1 and frame_count > frame_end):
                continue
            
            # Process the frame with MediaPipe
            image, results = mediapipe_detection(image, holistic)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            
            # Save keypoints
            np.save(os.path.join(output_dir, f"{saved_frames}.npy"), keypoints)
            saved_frames += 1
            
            # Save annotated image (optional)
            cv2.imwrite(os.path.join(output_dir, f"{saved_frames}.jpg"), image)
    
    cap.release()
    print(f"Processing complete. Saved {saved_frames} frames to {output_dir}")
    
    # Return path to the first saved frame for visualization
    if saved_frames > 0:
        return os.path.join(output_dir, "0.npy")
    return None

def visualize_demo():
    """Run a demonstration of the visualization tools."""
    from visualize import visualize_landmarks, visualize_keypoints
    
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VIDEO_PATH = os.path.join(BASE_DIR, "archive", "videos")
    
    # Get list of available videos
    available_videos = [f for f in os.listdir(VIDEO_PATH) if f.endswith('.mp4')]
    
    if not available_videos:
        print("No videos found in the archive/videos directory.")
        return
    
    # Use the first video for demonstration
    video_path = os.path.join(VIDEO_PATH, available_videos[0])
    print(f"Visualizing landmarks for {video_path}")
    
    # Visualize the video with landmarks
    visualize_landmarks(video_path)
    
    # Process the video and visualize extracted keypoints
    keypoints_path = process_single_video(os.path.splitext(available_videos[0])[0])
    
    if keypoints_path and os.path.exists(keypoints_path):
        print(f"Visualizing keypoints from {keypoints_path}")
        visualize_keypoints(keypoints_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Recognition Demo")
    parser.add_argument("--demo", action="store_true", help="Run visualization demo")
    parser.add_argument("--process", help="Process a specific video (e.g., '00335')")
    
    args = parser.parse_args()
    
    if args.demo:
        visualize_demo()
    elif args.process:
        process_single_video(args.process)
    else:
        # Display instructions if no arguments provided
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        VIDEO_PATH = os.path.join(BASE_DIR, "archive", "videos")
        available_videos = [os.path.splitext(f)[0] for f in os.listdir(VIDEO_PATH) if f.endswith('.mp4')]
        
        print("Sign Language Recognition Demo")
        print("-------------------------------")
        print("\nAvailable videos:", available_videos[:5], "...")
        print("\nOptions:")
        print("  --demo      Run a visualization demo using the first available video")
        print("  --process   Process a specific video by ID (e.g., --process 00335)")
        print("\nExamples:")
        print(f"  python demo.py --demo")
        print(f"  python demo.py --process {available_videos[0] if available_videos else '00335'}")
