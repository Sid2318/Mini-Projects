import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define emotion labels
EMOTIONS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

def show_emotion_examples():
    """
    Display examples of each emotion class from the FER2013 dataset
    """
    # Load the dataset
    csv_path = "archive/fer2013.csv"
    print("Loading dataset from:", csv_path)
    df = pd.read_csv(csv_path)
    
    # Create a figure to display examples
    plt.figure(figsize=(14, 14))
    
    # For each emotion class
    for emotion_idx in range(7):
        # Filter rows with this emotion
        emotion_df = df[df['emotion'] == emotion_idx]
        
        # Take 3 examples if available
        num_examples = min(3, len(emotion_df))
        
        for i in range(num_examples):
            # Get the pixel values and convert to image
            if i < len(emotion_df):
                pixels = emotion_df.iloc[i]['pixels']
                face = np.array([int(pixel) for pixel in pixels.split(' ')], dtype='uint8').reshape((48, 48))
                
                # Calculate subplot position
                subplot_idx = emotion_idx * 3 + i + 1
                plt.subplot(7, 3, subplot_idx)
                plt.imshow(face, cmap='gray')
                
                # Add title with emotion name
                if i == 0:
                    title = f"{emotion_idx}: {EMOTIONS[emotion_idx]}"
                else:
                    title = f"Example {i+1}"
                    
                plt.title(title)
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('emotion_class_examples.png')
    plt.show()
    print("Examples saved to 'emotion_class_examples.png'")
    
    # Display example descriptions
    print("\nEmotion Class Descriptions:")
    print("0: Angry - Furrowed brows, narrowed eyes, tight lips")
    print("1: Disgust - Wrinkled nose, raised upper lip, lowered brow")
    print("2: Fear - Raised eyebrows, wide open eyes, open mouth")
    print("3: Happy - Smile, raised cheeks, crinkled eyes")
    print("4: Sad - Drooping eyelids, downturned mouth, raised inner eyebrows")
    print("5: Surprise - Raised eyebrows, wide eyes, open mouth")
    print("6: Neutral - Relaxed facial features, no distinctive expressions")

if __name__ == "__main__":
    show_emotion_examples()
