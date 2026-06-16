import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

dataset_path = "dataset"

X = []
y = []

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            emotion_code = file.split("-")[2]
            emotion = emotion_dict[emotion_code]

            audio, sr = librosa.load(file_path, duration=3, offset=0.5)

            mfcc = np.mean(
                librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T,
                axis=0
            )

            X.append(mfcc)
            y.append(emotion)

X = np.array(X)

encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    Dense(256, activation='relu', input_shape=(40,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=50, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Accuracy: {accuracy*100:.2f}%")
