"""
Simple script to visualize a video with MediaPipe landmarks
"""

import cv2
import numpy as np
import os
import argparse
import mediapipe as mp

# MediaPipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def visualize_video(video_path, output_path=None):
    """
    Process a video file and show MediaPipe landmarks
    
    Args:
        video_path: Path to the input video file
        output_path: Optional path to save the output video
    """
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Video writer for saving output
    out = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize MediaPipe Holistic
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        frame_count = 0
        success = True
        
        while success:
            success, image = cap.read()
            if not success:
                break
            
            frame_count += 1
            print(f"\rProcessing frame: {frame_count}", end='')
            
            # Convert to RGB and process with MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            # Draw pose
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Draw face
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            
            # Draw left hand
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
            
            # Draw right hand
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
            
            # Add frame info
            cv2.putText(image, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display image
            cv2.imshow('MediaPipe Holistic', image)
            
            # Write to output file if specified
            if out:
                out.write(image)
            
            # Exit if ESC pressed
            if cv2.waitKey(5) & 0xFF == 27:
                break
    
    print(f"\nProcessed {frame_count} frames")
    
    # Release resources
    cap.release()
    if out:
        out.release()
        print(f"Output saved to: {output_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize video with MediaPipe")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", help="Path to save output video (optional)")
    
    args = parser.parse_args()
    
    visualize_video(args.video, args.output)
