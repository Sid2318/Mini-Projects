import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense, Dropout, Conv2D, MaxPooling2D, MaxPool2D, Flatten, 
    BatchNormalization, Input, Activation, GlobalAveragePooling2D
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.simplefilter("ignore")
# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

def load_and_preprocess_data(csv_path, use_sample=False, sample_size=5000):
    """
    Load and preprocess the FER2013 dataset.
    """
    print("Loading dataset from:", csv_path)
    df = pd.read_csv(csv_path)
    
    # Check the dataset
    print("Dataset shape:", df.shape)
    print("Dataset columns:", df.columns)
    print("Emotion distribution:")
    print(df['emotion'].value_counts())
    
    # 3.1 Shuffling the data
    df = df.sample(frac=1)
    
    # Extract features and labels
    pixels = df['pixels'].tolist()
    emotions = df['emotion'].values
    
    # If sampling is requested, take a subset of the data
    if use_sample:
        print(f"Using a sample of images for faster processing")
        if sample_size < len(pixels):
            indices = np.random.choice(len(pixels), sample_size, replace=False)
            pixels = [pixels[i] for i in indices]
            emotions = emotions[indices]
    
    # 3.2 One Hot Encoding
    emotions_onehot = to_categorical(emotions, num_classes=7)
    
    # Convert pixels to numpy arrays (changing image pixels to numpy array)
    train_pixels = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.array(face, dtype='uint8').reshape((48, 48, 1))
        # Convert grayscale to RGB for MobileNet (which expects 3 channels)
        face_rgb = np.repeat(face, 3, axis=-1)  # Repeat the grayscale channel 3 times
        train_pixels.append(face_rgb)
    
    train_pixels = np.array(train_pixels)
    
    # 3.3 Preprocess for MobileNet
    # Use MobileNet's specific preprocessing function instead of simple normalization
    train_pixels = preprocess_input(train_pixels)
    
    # 3.5 Train test validation split
    # Split 10% of data for testing, then 10% of remaining for validation
    X_train, X_test, y_train, y_test = train_test_split(
        train_pixels, emotions_onehot, test_size=0.1, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, shuffle=False
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 3.6 Data augmentation using ImageDataGenerator
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    valgen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    datagen.fit(X_train)
    valgen.fit(X_val)
    
    train_generator = datagen.flow(X_train, y_train, batch_size=64)
    val_generator = valgen.flow(X_val, y_val, batch_size=64)
    
    # Create a data object with all necessary information
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'train_generator': train_generator,
        'val_generator': val_generator
    }
    
    return data

def create_model():
    """
    Create a model using pre-trained ResNet50 with frozen convolutional layers
    and custom dense layers for emotion classification.
    """
    # Load the pre-trained ResNet50 model without the classification layers
    # Input shape: 48x48x3 (RGB images scaled up from grayscale)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(48, 48, 3)
    )
    
    # Freeze the convolutional layers in ResNet50
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create a new model
    inputs = Input(shape=(48, 48, 3))
    x = base_model(inputs, training=False)
    
    # Add custom classification layers
    x = GlobalAveragePooling2D()(x)
    
    # First dense block with more units for ResNet's larger feature maps
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Second dense block
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer for the 7 emotion classes
    outputs = Dense(7, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Display a summary of the model architecture
    print("ResNet50 base model summary (frozen):")
    base_model.summary()
    
    # Compile the model with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, data, epochs=50, batch_size=64):
    """
    Train the model with data augmentation.
    4.2 Early stopping implementation
    """
    # Create callbacks - Early Stopping and ModelCheckpoint
    checkpointer = [
        EarlyStopping(
            monitor='val_accuracy', 
            verbose=1,
            restore_best_weights=True,
            mode="max",
            patience=10
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            mode="max"
        )
    ]
    
    # Train the model
    print("Training model with data augmentation...")
    history = model.fit(
        data['train_generator'],
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=checkpointer,
        validation_data=data['val_generator']
    )
    
    return model, history

def plot_training_history(history):
    """
    Plot the training and validation accuracy and loss.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model, data):
    """
    Evaluate the model on the test set and show detailed metrics.
    4.3 Visualizing results
    """
    # Evaluate the model
    loss, accuracy = model.evaluate(data['X_test'], data['y_test'])
    print(f"Test Acc: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")
    
    # Make predictions
    preds = model.predict(data['X_test'])
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(data['y_test'], axis=1)
    
    # Create class labels
    CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    
    # Visualize examples of predictions
    figure = plt.figure(figsize=(20, 8))
    for i, index in enumerate(np.random.choice(data['X_test'].shape[0], size=24, replace=False)):
        ax = figure.add_subplot(4, 6, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(data['X_test'][index]))
        predict_index = label_dict[y_pred[index]]
        true_index = label_dict[y_true[index]]
        
        ax.set_title("{} ({})".format(
            predict_index, 
            true_index),
            color=("green" if predict_index == true_index else "red")
        )
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.show()
    
    # Create confusion matrix
    cm_data = confusion_matrix(y_true, y_pred)
    cm = pd.DataFrame(cm_data, columns=CLASS_LABELS, index=CLASS_LABELS)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    plt.figure(figsize=(15, 10))
    plt.title('Confusion Matrix', fontsize=20)
    sns.set(font_scale=1.2)
    ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print("Classification Report:")
    report = classification_report(y_true, y_pred, 
                                 target_names=CLASS_LABELS,
                                 digits=3)
    print(report)
    
    # Calculate accuracy per class
    print("\nResults per emotion class:")
    for emotion_idx in range(7):
        # Get indices for this emotion
        indices = np.where(y_true == emotion_idx)[0]
        if len(indices) > 0:
            correct = np.sum(y_pred[indices] == emotion_idx)
            accuracy = correct / len(indices)
            print(f"{CLASS_LABELS[emotion_idx]}: {accuracy:.4f} accuracy ({correct}/{len(indices)})")

def show_examples(model, data):
    """
    Show examples of each emotion class with model predictions.
    """
    # Make predictions on test data
    predictions = model.predict(data['X_test'])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(data['y_test'], axis=1)
    
    # Define emotion labels dictionary
    label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    
    plt.figure(figsize=(15, 23))
    
    for emotion_idx in range(7):
        # Get indices for this emotion
        indices = np.where(true_classes == emotion_idx)[0]
        
        if len(indices) > 0:
            # Show one example for each emotion
            idx = indices[0]
            image = data['X_test'][idx]
            
            # Calculate subplot position
            plt.subplot(1, 7, emotion_idx + 1)
            
            # Show the image (grayscale)
            plt.imshow(image.squeeze(), cmap='gray')
            
            # Get the predicted label
            pred_class = predicted_classes[idx]
            
            # Add title with emotion name and prediction
            title = f"{label_dict[emotion_idx]}"
            if pred_class != emotion_idx:
                title += f"\nPred: {label_dict[pred_class]}"
            
            # Color based on correct/incorrect prediction
            color = 'green' if pred_class == emotion_idx else 'red'
            plt.title(title, color=color)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('emotion_examples.png')
    plt.show()

if __name__ == "__main__":
    # Path to the dataset
    csv_path = "archive/fer2013.csv"
    
    # Determine if we should use a sample (for faster experimentation)
    # Set use_sample to False for full training
    use_sample = True  # Set to False for full training
    sample_size = 8000  # Number of samples to use if use_sample is True
    
    # Load and preprocess data
    print("\n===== Loading and preprocessing data =====")
    data = load_and_preprocess_data(csv_path, use_sample=use_sample, sample_size=sample_size)
    
    # Create model with ResNet50 and custom dense layers
    print("\n===== Creating ResNet50 transfer learning model =====")
    model = create_model()
    model.summary()
    
    # Set the number of epochs - use fewer epochs for transfer learning
    # as the model will converge faster
    epochs = 20
    
    # Train model
    print("\n===== Training the model =====")
    print("Using transfer learning with frozen ResNet50 layers...")
    model, history = train_model(model, data, epochs=epochs)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\n===== Evaluating the model =====")
    evaluate_model(model, data)
    
    # Show examples
    print("\n===== Showing emotion examples =====")
    show_examples(model, data)
    
    # Save the final model
    model.save('emotion_detection_resnet_model.h5')
    print("\nModel saved successfully as 'emotion_detection_resnet_model.h5'!")
    
    # Optional: Fine-tune the model by unfreezing some of the top layers
    fine_tune_model = False  # Set to True if you want to fine-tune the model
    
    if fine_tune_model:
        print("\n===== Fine-tuning the model by unfreezing top layers =====")
        # Unfreeze the top layers of the ResNet50 model (the last conv block)
        for layer in model.layers[0].layers[-30:]:
            layer.trainable = True
        
        # Recompile the model with a lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.00001),  # Lower learning rate for fine-tuning
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for a few more epochs
        model, history_fine_tune = train_model(model, data, epochs=10)
        
        # Evaluate fine-tuned model
        print("\n===== Evaluating the fine-tuned model =====")
        evaluate_model(model, data)
        
        # Save the fine-tuned model
        model.save('emotion_detection_resnet_finetuned.h5')
        print("\nFine-tuned model saved as 'emotion_detection_resnet_finetuned.h5'!")
