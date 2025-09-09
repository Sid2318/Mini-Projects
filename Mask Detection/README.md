# Face Mask Detection with VGG16 and Streamlit

## Overview

This project is a complete pipeline for face mask detection using deep learning and transfer learning (VGG16). It includes:

- Data preprocessing and caching
- Model training with data augmentation and early stopping
- Model evaluation and saving
- Real-time mask detection using a webcam via a Streamlit app

## Project Structure

```
Mask Detection/
├── archive (1)/data/
│   ├── with_mask/         # Images of people with masks
│   └── without_mask/      # Images of people without masks
├── processed_data/        # Cached numpy arrays for fast reload
│   ├── mask_data.pkl
│   ├── mask_labels.pkl
│   └── train_test_split.pkl
├── main.py                # Data processing, model training, saving
├── improved_app.py        # Streamlit app for real-time mask detection
├── mask_detector_vgg16.h5 # Saved Keras model
├── best_mask_model.h5     # Best model (from early stopping)
├── cnn_env/               # Python virtual environment (optional)
```

## Setup Instructions

1. **Install dependencies** (preferably in a virtual environment):
   ```bash
   pip install tensorflow opencv-python opencv-contrib-python streamlit matplotlib scikit-learn numpy
   ```
2. **Prepare the dataset**:

   - Place your images in `archive (1)/data/with_mask/` and `archive (1)/data/without_mask/`.

3. **Train the model**:

   ```bash
   python main.py
   ```

   - This will preprocess data, train the model with VGG16, and save `mask_detector_vgg16.h5` and `best_mask_model.h5`.

4. **Run the Streamlit app**:
   ```bash
   streamlit run improved_app.py
   ```
   - The app will use your webcam, detect faces, and classify mask/no mask in real time.

## Features

- **Data Caching**: Preprocessed data is saved for fast reloads.
- **Transfer Learning**: Uses VGG16 as a feature extractor.
- **Data Augmentation**: Improves generalization to real-world images.
- **Early Stopping**: Prevents overfitting and saves the best model.
- **Model Evaluation**: Prints test accuracy after training.
- **Webcam App**: Real-time mask detection with face detection (DNN or Haar cascade).

## Customization

- Adjust `IMG_SIZE` in `main.py` and `improved_app.py` if you want to use a different input size.
- You can further tune the model architecture, augmentation, or training parameters in `main.py`.

## Troubleshooting

- If the webcam app is slow or inaccurate, try increasing training epochs or improving your dataset.
- For best face detection, ensure you have `opencv-contrib-python` installed for DNN support.
- If you encounter errors, check that all dependencies are installed and your dataset is correctly structured.

## Credits

- VGG16: [Keras Applications](https://keras.io/api/applications/vgg/)
- Face Detection: OpenCV DNN and Haar Cascade
- Streamlit: [streamlit.io](https://streamlit.io/)

---

**Author:** Siddhi Mohol
