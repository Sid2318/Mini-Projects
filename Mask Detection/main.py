import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# ---------------- Configuration ----------------
IMG_SIZE = 100
DATA_DIR = "processed_data"
DATA_FILE = os.path.join(DATA_DIR, "mask_data.pkl")
LABELS_FILE = os.path.join(DATA_DIR, "mask_labels.pkl")
SPLIT_FILE = os.path.join(DATA_DIR, "train_test_split.pkl")
NUM_CLASSES = 1  # Binary classification: with_mask / without_mask

# ---------------- Helper Functions ----------------
def setup_data_directory():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")


def display_dataset_info():
    base_dir = "archive (1)/data"
    with_mask_dir = os.path.join(base_dir, "with_mask")
    without_mask_dir = os.path.join(base_dir, "without_mask")
    print("With mask images:", len(os.listdir(with_mask_dir)))
    print("Without mask images:", len(os.listdir(without_mask_dir)))
    return base_dir, with_mask_dir, without_mask_dir


def show_sample_images(with_mask_dir, without_mask_dir):
    plt.figure(figsize=(10,5))
    for i, category in enumerate([with_mask_dir, without_mask_dir]):
        file = os.listdir(category)[0]
        img_path = os.path.join(category, file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1,2,i+1)
        plt.imshow(img)
        plt.title("With Mask" if i==0 else "Without Mask")
        plt.axis("off")
    plt.show()


def compare_specific_images(img1_path, img2_path):
    plt.figure(figsize=(10,5))
    img1 = cv2.imread(img1_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(img2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.title("With Mask")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.title("Without Mask")
    plt.axis("off")
    plt.show()


def process_data(with_mask_dir, without_mask_dir):
    if os.path.exists(DATA_FILE) and os.path.exists(LABELS_FILE):
        print("Loading processed data from cache...")
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
        with open(LABELS_FILE, 'rb') as f:
            labels = pickle.load(f)
        return data, labels

    print("Processing images...")
    data, labels = [], []

    for category, label in zip([with_mask_dir, without_mask_dir], [0, 1]):
        print(f"Processing {'with mask' if label==0 else 'without mask'} images...")
        for file in os.listdir(category):
            img_path = os.path.join(category, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(label)

    data = np.array(data)/255.0
    labels = np.array(labels)

    with open(DATA_FILE, 'wb') as f:
        pickle.dump(data, f)
    with open(LABELS_FILE, 'wb') as f:
        pickle.dump(labels, f)
    print(f"Processed data saved to {DATA_DIR}")
    return data, labels


def split_dataset(data, labels):
    if os.path.exists(SPLIT_FILE):
        print("Loading train/test split from cache...")
        with open(SPLIT_FILE, 'rb') as f:
            split_data = pickle.load(f)
        return split_data['X_train'], split_data['X_test'], split_data['y_train'], split_data['y_test']

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    split_data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
    with open(SPLIT_FILE, 'wb') as f:
        pickle.dump(split_data, f)

    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


# ---------------- Model Function ----------------

def build_and_train_vgg16_model(X_train, y_train, X_test, y_test):
    print("Building VGG16-based model...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
        
    # Add data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),  # Increased neurons
        Dropout(0.5),
        Dense(128, activation='relu'),  # Additional layer
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    # Add early stopping callback
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=5,          # Stop after 5 epochs without improvement
        verbose=1,           # Print messages
        restore_best_weights=True  # Restore model weights from the epoch with the best value of monitored metric
    )
    
    # Model checkpoint to save the best model
    checkpoint = ModelCheckpoint(
        'best_mask_model.h5',  # Path to save the best model
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    # Train for more epochs with early stopping
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=20,  # More epochs (early stopping will prevent overfitting)
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping, checkpoint]  # Add callbacks
    )
    
    # Evaluate model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {scores[1]*100:.2f}%")
    
    # Save the model
    model.save('mask_detector_vgg16.h5')
    print("Model saved as mask_detector_vgg16.h5")
    return model, history

# Predict function for a single image (numpy array, RGB, 0-1, shape (IMG_SIZE, IMG_SIZE, 3))
def predict_mask(model, img_arr):
    import numpy as np
    img = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    return 'Mask' if pred[0][0] < 0.5 else 'No Mask', float(pred[0][0])


# ---------------- Main Workflow ----------------
def main():
    setup_data_directory()
    base_dir, with_mask_dir, without_mask_dir = display_dataset_info()
    show_sample_images(with_mask_dir, without_mask_dir)

    data, labels = process_data(with_mask_dir, without_mask_dir)
    X_train, X_test, y_train, y_test = split_dataset(data, labels)

    model = build_and_train_vgg16_model(X_train, y_train, X_test, y_test)
    preds = []
    for img in X_test:
        label, confidence = predict_mask(model, img)
        preds.append((label, confidence))
    print(preds[:10])  # Print first 10 predictions
    return model


if __name__ == "__main__":
    model = main()
