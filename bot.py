from scipy.io.wavfile import read
from IPython.display import Audio, display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
import os
import pickle

dataset = "dataset/splitted/"
num_labels = 10

labels = []
audios = []
for label in range(num_labels):
    label_path = f"{dataset}/{label}"
    for file in sorted(os.listdir(label_path)):
        file_path = label_path + "/" + file
        sample_rate, audio = read(file_path)
        labels.append(label)
        audios.append(audio)
labels = np.array(labels)

max_duration_sec = 0.6
max_duration = int(max_duration_sec * sample_rate + 1e-6)

features = []
features_flatten = []
for audio in audios:
    if len(audio) < max_duration:
        audio = np.pad(audio, (0, max_duration - len(audio)), constant_values=0)
    feature = librosa.feature.melspectrogram(audio.astype(float), sample_rate, n_mels=128, fmax=16000)
    features.append(feature)
    features_flatten.append(feature.reshape(-1))

features_train, features_test, labels_train, labels_test = train_test_split(features_flatten, labels)

#model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
model = MLPClassifier(hidden_layer_sizes=(300))

model.fit(X=features_train, y=labels_train)

labels_test_predicted = model.predict(X=features_test)

print((labels_test_predicted == labels_test).mean())

labels_test_predicted

filename = "model.pkl"
model_pickled = pickle.dumps(model)
with open(filename, 'wb') as f:
    f.write(model_pickled)
