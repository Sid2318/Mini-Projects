import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from main import load_and_preprocess_data, EMOTIONS

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def train_test_model():
    """
    Loads the trained model and evaluates it on test data.
    """
    # Check if the model exists
    model_path = 'emotion_detection_model.h5'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please run main.py first to train the model.")
        return
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Load and preprocess test data
    csv_path = "archive/fer2013.csv"
    _, _, _, _, test_images, test_emotions = load_and_preprocess_data(csv_path)
    
    # Evaluate the model
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_images, test_emotions)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Make predictions
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate and display confusion matrix
    cm = confusion_matrix(test_emotions, predicted_classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Set labels
    emotion_names = [EMOTIONS[i] for i in range(7)]
    tick_marks = np.arange(len(emotion_names))
    plt.xticks(tick_marks, emotion_names, rotation=45)
    plt.yticks(tick_marks, emotion_names)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('test_confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print("Classification Report:")
    report = classification_report(test_emotions, predicted_classes, 
                                  target_names=emotion_names)
    print(report)
    
    # Show examples for each emotion with predictions
    show_emotion_examples(test_images, test_emotions, predicted_classes)
    
    return model, test_images, test_emotions, predicted_classes

def show_emotion_examples(test_images, test_emotions, predicted_classes):
    """
    Show examples of each emotion class (0-6) with their predictions.
    """
    plt.figure(figsize=(18, 12))
    
    # For each emotion class
    for emotion_idx in range(7):
        # Find examples where the true emotion is the current emotion
        indices = np.where(test_emotions == emotion_idx)[0]
        
        if len(indices) > 0:
            # Take the first 3 examples (or fewer if not enough available)
            num_examples = min(3, len(indices))
            
            for i in range(num_examples):
                idx = indices[i]
                image = test_images[idx]
                
                # Calculate subplot position
                subplot_idx = emotion_idx * 3 + i + 1
                plt.subplot(7, 3, subplot_idx)
                plt.imshow(image[:, :, 0], cmap='gray')
                
                # Add title with emotion name and prediction
                true_emotion = EMOTIONS[emotion_idx]
                pred_emotion = EMOTIONS[predicted_classes[idx]]
                
                if true_emotion == pred_emotion:
                    title = f"{true_emotion}\nCorrect!"
                    color = 'green'
                else:
                    title = f"{true_emotion}\nPredicted: {pred_emotion}"
                    color = 'red'
                
                plt.title(title, color=color)
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('emotion_class_examples.png')
    plt.show()

def analyze_misclassifications(test_images, test_emotions, predicted_classes):
    """
    Analyze and visualize common misclassifications between emotion classes.
    """
    # Create a figure to show common misclassifications
    plt.figure(figsize=(15, 10))
    
    # Track misclassification pairs we've shown
    shown_pairs = set()
    subplot_idx = 1
    
    # For each emotion pair
    for true_emotion in range(7):
        for pred_emotion in range(7):
            if true_emotion != pred_emotion:
                # Find examples where true_emotion was misclassified as pred_emotion
                indices = np.where((test_emotions == true_emotion) & 
                                   (predicted_classes == pred_emotion))[0]
                
                # Check if we have any misclassifications and haven't shown this pair yet
                pair_key = f"{true_emotion}_{pred_emotion}"
                if len(indices) > 0 and pair_key not in shown_pairs:
                    # Take the first example
                    idx = indices[0]
                    image = test_images[idx]
                    
                    # Only show up to 12 misclassification examples
                    if subplot_idx <= 12:
                        plt.subplot(3, 4, subplot_idx)
                        plt.imshow(image[:, :, 0], cmap='gray')
                        plt.title(f"True: {EMOTIONS[true_emotion]}\nPredicted: {EMOTIONS[pred_emotion]}")
                        plt.axis('off')
                        
                        # Increment subplot index and mark this pair as shown
                        subplot_idx += 1
                        shown_pairs.add(pair_key)
    
    plt.tight_layout()
    plt.savefig('common_misclassifications.png')
    plt.show()

if __name__ == "__main__":
    model, test_images, test_emotions, predicted_classes = train_test_model()
    
    # Analyze common misclassifications
    analyze_misclassifications(test_images, test_emotions, predicted_classes)
    
    # Print summary of results per emotion class
    emotion_names = [EMOTIONS[i] for i in range(7)]
    print("\nResults per emotion class:")
    for emotion_idx in range(7):
        # Get indices for this emotion
        indices = np.where(test_emotions == emotion_idx)[0]
        correct = np.sum(predicted_classes[indices] == emotion_idx)
        accuracy = correct / len(indices) if len(indices) > 0 else 0
        
        print(f"{EMOTIONS[emotion_idx]}: {accuracy:.4f} accuracy ({correct}/{len(indices)})")
