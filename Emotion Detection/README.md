# Facial Emotion Detection with CNN

This project implements a facial emotion detection system using a custom Convolutional Neural Network (CNN) architecture. It classifies facial expressions into 7 emotion categories:

0. Angry
1. Disgust
2. Fear
3. Happy
4. Sad
5. Surprise
6. Neutral

## Dataset

The project uses the FER2013 dataset which contains 48x48 pixel grayscale images of faces labeled with one of the 7 emotions.

## Requirements

- Python 3.6+
- TensorFlow 2.x
- pandas
- matplotlib
- scikit-learn

You can install the required dependencies with:

```
pip install tensorflow pandas matplotlib scikit-learn
```

## Project Structure

- `main.py`: Contains the code to load the dataset, create the model architecture, train the model, and evaluate it.
- `train_test.py`: Contains code for more detailed analysis of the trained model, showing examples of each emotion class, and analyzing misclassifications.

## How to Run

1. First, run the main script to preprocess the data, train the model, and save it:

```
python main.py
```

2. After training, you can run the train_test script for more detailed analysis:

```
python train_test.py
```

## Model Architecture

The model uses a custom CNN architecture with multiple convolutional blocks:

1. First block: 2 Conv2D layers (32 filters) with BatchNormalization, MaxPooling and Dropout
2. Second block: 2 Conv2D layers (64 filters) with BatchNormalization, MaxPooling and Dropout
3. Third block: 2 Conv2D layers (128 filters) with BatchNormalization, MaxPooling and Dropout
4. Flatten layer
5. Dense layer with 512 units, BatchNormalization and Dropout
6. Dense layer with 256 units, BatchNormalization and Dropout
7. Output layer with 7 units (for the 7 emotions) and softmax activation

## Training Process

1. First phase: Train only the top layers with the base model frozen
2. Second phase: Fine-tune by unfreezing the last 20 layers of the base model

## Output Examples

The model will generate several visualization outputs:

- `training_history.png`: Plot of training and validation accuracy and loss
- `confusion_matrix.png`: Confusion matrix of the model's predictions
- `emotion_examples.png`: Examples of each emotion class with predictions
- `common_misclassifications.png`: Common misclassification examples

## Emotion Class Examples

The script will display examples of each emotion class (0-6):

0. Angry: Furrowed brows, narrowed eyes, tight lips
1. Disgust: Wrinkled nose, raised upper lip, lowered brow
2. Fear: Raised eyebrows, wide open eyes, open mouth
3. Happy: Smile, raised cheeks, crinkled eyes
4. Sad: Drooping eyelids, downturned mouth, raised inner eyebrows
5. Surprise: Raised eyebrows, wide eyes, open mouth
6. Neutral: Relaxed facial features, no distinctive expressions
