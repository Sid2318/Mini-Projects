import cv2
import mediapipe as mp
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# MediaPipe models and utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def visualize_landmarks(video_path, output_path=None):
    """
    Visualize MediaPipe landmarks on a video and save the result
    
    Args:
        video_path: Path to the input video
        output_path: Path to save the output video (optional)
    """
    # Create output directory if specified
    if output_path and not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output path is specified
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize MediaPipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame_count = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # Convert the image to RGB and process it with MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            # Draw landmarks on the image
            annotated_image = image.copy()
            
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Draw face landmarks
            mp_drawing.draw_landmarks(
                annotated_image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            
            # Draw left hand landmarks
            mp_drawing.draw_landmarks(
                annotated_image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
            
            # Draw right hand landmarks
            mp_drawing.draw_landmarks(
                annotated_image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
            
            # Add frame number
            cv2.putText(annotated_image, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write to output video if specified
            if output_path:
                out.write(annotated_image)
            
            # Display the frame
            cv2.imshow('MediaPipe Holistic', annotated_image)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                break
            
            frame_count += 1
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames")
    if output_path:
        print(f"Output video saved to {output_path}")

def visualize_keypoints(npy_file_path):
    """
    Visualize keypoints from a saved numpy file
    
    Args:
        npy_file_path: Path to the .npy file containing keypoints
    """
    # Load the keypoints
    keypoints = np.load(npy_file_path)
    
    # Extract different parts of the keypoints
    pose = keypoints[:33*4].reshape(33, 4)
    face = keypoints[33*4:33*4+468*3].reshape(468, 3)
    left_hand = keypoints[33*4+468*3:33*4+468*3+21*3].reshape(21, 3)
    right_hand = keypoints[33*4+468*3+21*3:].reshape(21, 3)
    
    # Create a 2D plot to visualize the landmarks
    plt.figure(figsize=(10, 10))
    
    # Plot pose landmarks
    plt.subplot(2, 2, 1)
    plt.scatter(pose[:, 0], -pose[:, 1], c='blue', alpha=0.5)
    plt.title('Pose Landmarks')
    plt.xlim(0, 1)
    plt.ylim(-1, 0)
    
    # Plot face landmarks
    plt.subplot(2, 2, 2)
    plt.scatter(face[:, 0], -face[:, 1], c='green', alpha=0.5, s=1)
    plt.title('Face Landmarks')
    plt.xlim(0, 1)
    plt.ylim(-1, 0)
    
    # Plot left hand landmarks
    plt.subplot(2, 2, 3)
    plt.scatter(left_hand[:, 0], -left_hand[:, 1], c='red', alpha=0.5)
    plt.title('Left Hand Landmarks')
    plt.xlim(0, 1)
    plt.ylim(-1, 0)
    
    # Plot right hand landmarks
    plt.subplot(2, 2, 4)
    plt.scatter(right_hand[:, 0], -right_hand[:, 1], c='purple', alpha=0.5)
    plt.title('Right Hand Landmarks')
    plt.xlim(0, 1)
    plt.ylim(-1, 0)
    
    plt.tight_layout()
    plt.suptitle(f'Keypoints from {os.path.basename(npy_file_path)}')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MediaPipe landmarks")
    parser.add_argument("--video", help="Path to the input video")
    parser.add_argument("--output", help="Path to save the output video (optional)")
    parser.add_argument("--keypoints", help="Path to the .npy file containing keypoints (optional)")
    
    args = parser.parse_args()
    
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file '{args.video}' not found")
            print(f"Example usage: python visualize.py --video archive/videos/00335.mp4 --output output.mp4")
        else:
            visualize_landmarks(args.video, args.output)
    elif args.keypoints:
        if not os.path.exists(args.keypoints):
            print(f"Error: Keypoints file '{args.keypoints}' not found")
            print(f"Example usage: python visualize.py --keypoints MP_Data_Frames/book/00335/0.npy")
        else:
            visualize_keypoints(args.keypoints)
    else:
        print("Please specify either --video or --keypoints")
        print("\nExamples:")
        print("  python visualize.py --video archive/videos/00335.mp4")
        print("  python visualize.py --video archive/videos/00335.mp4 --output output.mp4")
        print("  python visualize.py --keypoints MP_Data_Frames/book/00335/0.npy")
