# Sign Language Recognition with MediaPipe

This project processes sign language videos from the WLASL dataset, extracts features using MediaPipe, and trains an LSTM model for sign language recognition.

## Setup

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Project structure:

- `main.py`: Main script for processing videos and training the model
- `archive/`: Dataset folder containing videos and metadata
- `MP_Data_Frames/`: Output folder for processed data (will be created)
- `Logs/`: TensorBoard logs for model training (will be created)

## Steps

The `main.py` script performs the following steps:

1. Processes videos from the dataset and extracts landmarks using MediaPipe
2. Lists the processed data structure
3. Prepares training data from extracted features
4. Trains an LSTM model for sign language recognition

## Running the Code

Simply execute:

```
python main.py
```

## About the Dataset

The WLASL (Word-Level American Sign Language) dataset contains videos of ASL signs for various words. Each video shows a person performing a sign language gesture. The dataset includes:

- Videos in MP4 format
- Metadata in JSON format with labels and timing information
- A class list of all sign words

## Feature Extraction

The code extracts the following features using MediaPipe:

- Pose landmarks (33 points with x, y, z, visibility)
- Face landmarks (468 points with x, y, z)
- Left hand landmarks (21 points with x, y, z)
- Right hand landmarks (21 points with x, y, z)

These features are used to train an LSTM model for recognizing sign language gestures.
