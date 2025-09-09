import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import os

label_map = {"HELLO":0, "THANKYOU":1}

X, y = [], []

for word, label in label_map.items():
    folder = f"sign_data/{word}"
    for file in os.listdir(folder):
        seq = np.load(os.path.join(folder, file))
        X.append(seq)
        y.append(label)

X = np.array(X)
y = np.array(y)

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(64))
model.add(Dense(len(label_map), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=4)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/sign_lstm.h5")
