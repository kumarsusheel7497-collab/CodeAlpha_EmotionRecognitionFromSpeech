# Emotion Recognition From Speech

## Objective
Recognize human emotions from speech audio using Machine Learning.

## Dataset
RAVDESS Emotional Speech Dataset

## Features
- MFCC Feature Extraction
- Deep Learning Model
- Emotion Classification

## Technologies Used
- Python
- Librosa
- TensorFlow
- Scikit-Learn


## CODE
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

# Ensure the dataset directory exists
if not os.path.exists(dataset_path):
    print(f"Error: Dataset path '{dataset_path}' not found. Please ensure the directory exists and contains audio files.")
else:
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)

                try:
                    # Extract emotion code from filename (assuming RAVDESS-like format)
                    emotion_code = file.split("-")[2]
                    emotion = emotion_dict[emotion_code]

                    # Load audio file and extract MFCC features
                    audio, sr = librosa.load(file_path, duration=3, offset=0.5)

                    # Ensure mfcc has the same number of features as expected by the model (40)
                    mfcc = np.mean(
                        librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T,
                        axis=0
                    )

                    X.append(mfcc)
                    y.append(emotion)
                except Exception as e:
                    print(f"Could not process file {file_path}: {e}")

X = np.array(X)

# Check if any data was loaded before proceeding
if X.size == 0:
    print("No audio files were processed or found in the dataset. Please check the 'dataset' directory and file format.")
elif X.shape[0] < 2: # Check if there's enough data to split
    print(f"Only {X.shape[0]} audio file(s) found. Need at least 2 files to perform a train-test split and train the model. Please upload more files to the 'dataset' directory.")
else:
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

    print("\nStarting model training...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    print("Model training complete.")

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nAccuracy: {accuracy*100:.2f}%")

    # --- Prediction part ---
    # Define a sample audio path for prediction. You can change this to any .wav file.
    # For demonstration, we'll use the first file processed from the dataset if available.
    sample_audio_path = None
    if len(files) > 0: # Ensure 'files' list is not empty
        for f in files:
            if f.endswith(".wav"):
                sample_audio_path = os.path.join(dataset_path, f)
                break

    if sample_audio_path and os.path.exists(sample_audio_path):
        try:
            print(f"\nAttempting prediction for: {sample_audio_path}")
            sample_audio_raw, sample_sr = librosa.load(sample_audio_path, duration=3, offset=0.5)
            sample_audio_mfcc = np.mean(librosa.feature.mfcc(y=sample_audio_raw, sr=sample_sr, n_mfcc=40).T, axis=0)

            # Reshape the MFCC features to match the model's input (1 sample, 40 features)
            sample_audio = np.array([sample_audio_mfcc])

            prediction = model.predict(sample_audio)
            emotion = encoder.inverse_transform([np.argmax(prediction)])
            print("Predicted Emotion:", emotion[0])
        except Exception as e:
            print(f"Could not perform prediction for {sample_audio_path}: {e}")
    else:
        print("\nNo sample audio file available for prediction. Ensure 'dataset' has files and sample_audio_path is set correctly.")

      
##OUTPUT
/usr/local/lib/python3.12/dist-packages/keras/src/layers/core/dense.py:106: 
Starting model training...
Model training complete.

Accuracy: 60.00%

Attempting prediction for: dataset/03-01-02-02-01-01-01.wav
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 101ms/step
Predicted Emotion: calm


## Emotions Detected
- Happy
- Sad
- Angry
- Calm
- Fearful
- Neutral
- Disgust
- Surprised
