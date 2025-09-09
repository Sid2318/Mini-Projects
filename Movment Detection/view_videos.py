"""
Simple script to display videos from the dataset
"""

import cv2
import os
import argparse
import json

def list_available_videos(video_dir):
    """List available videos and their information"""
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print(f"Found {len(videos)} videos in {video_dir}")
    print(f"Examples: {videos[:5]}")
    return videos

def get_video_metadata(video_id, json_path):
    """Get metadata for a video from the JSON file"""
    try:
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        # Find the sign label for this video
        for item in metadata:
            for instance in item['instances']:
                if str(instance['video_id']) == video_id:
                    return {
                        'label': item['gloss'],
                        'frame_start': instance.get('frame_start', 1),
                        'frame_end': instance.get('frame_end', -1),
                        'fps': instance.get('fps', 30)
                    }
        
        return None
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return None

def play_video(video_path, info=None):
    """Play a video using OpenCV"""
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # If info is provided, set frame ranges
    frame_start = 1
    frame_end = -1
    if info:
        frame_start = info.get('frame_start', 1)
        frame_end = info.get('frame_end', -1)
        label = info.get('label', 'Unknown')
        print(f"Playing sign: '{label}' (Frames: {frame_start} to {frame_end if frame_end != -1 else 'end'})")
    
    # Play video
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames outside the specified range
        if frame_count < frame_start:
            continue
        if frame_end != -1 and frame_count > frame_end:
            break
        
        # Add frame info text
        if info and 'label' in info:
            cv2.putText(frame, f"Sign: {info['label']}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Frame: {frame_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Video', frame)
        
        # Exit if ESC pressed or q
        key = cv2.waitKey(25) & 0xFF
        if key == 27 or key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="View videos from the dataset")
    parser.add_argument("--list", action="store_true", help="List available videos")
    parser.add_argument("--video_id", help="ID of the video to play (e.g., '00335')")
    
    args = parser.parse_args()
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(base_dir, "archive", "videos")
    json_path = os.path.join(base_dir, "archive", "WLASL_v0.3.json")
    
    # Make sure video directory exists
    if not os.path.exists(video_dir):
        print(f"Error: Video directory not found: {video_dir}")
        return
    
    if args.list:
        list_available_videos(video_dir)
        return
    
    if args.video_id:
        video_path = os.path.join(video_dir, f"{args.video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            videos = list_available_videos(video_dir)
            return
        
        # Get metadata for the video
        metadata = None
        if os.path.exists(json_path):
            metadata = get_video_metadata(args.video_id, json_path)
        
        # Play the video
        play_video(video_path, metadata)
        return
    
    # If no arguments provided, list videos and prompt for input
    videos = list_available_videos(video_dir)
    
    if videos:
        print("\nTo play a video, use: --video_id VIDEO_ID")
        print(f"Example: python {os.path.basename(__file__)} --video_id {os.path.splitext(videos[0])[0]}")

if __name__ == "__main__":
    main()
