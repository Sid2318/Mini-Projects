# Troubleshooting Guide for MediaPipe

If you're experiencing issues with MediaPipe and the "MessageFactory" error, follow these steps to resolve them:

## Common Error

```
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
```

or

```
ImportError: cannot import name 'builder' from 'google.protobuf.internal'
```

## Solution 1: Downgrade MediaPipe and Protobuf

Create a new virtual environment and install compatible versions:

```powershell
# Create and activate virtual environment
python -m venv mp_venv
.\mp_venv\Scripts\Activate.ps1

# Install compatible versions
pip install mediapipe==0.8.9.1
pip install protobuf==3.19.1
pip install opencv-python==4.5.5.64
pip install numpy==1.22.3
pip install tensorflow==2.9.0
```

## Solution 2: Use Simple Video Viewer

For simply viewing the videos without MediaPipe:

```powershell
python view_videos.py --list
python view_videos.py --video_id 00335
```

## Alternative Version of Requirements

If you want to try different compatible versions:

```
opencv-python==4.5.5.64
mediapipe==0.8.9.1
protobuf==3.19.1
numpy==1.22.3
tensorflow==2.9.0
scikit-learn==1.0.2
matplotlib==3.5.1
```

## Next Steps

1. Make sure you have the right versions installed
2. Try using the view_videos.py script to view the dataset
3. If you still want to use MediaPipe, create a fresh virtual environment with the versions specified above
