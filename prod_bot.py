import sys          #argv
import numpy as np # generate random
import os          # get current directory path
import subprocess  # execute ffmpeg
import telebot     # run telegram bot
from scipy.io.wavfile import read, write
from IPython.display import Audio, display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import subprocess

import time
import matplotlib.pyplot as plt
import numpy as np
import librosa
#import torch
import os
from datetime import datetime # generate log
import pickle


filename = "model/model.pkl"
with open(filename, 'rb') as f:
    model_pickled = f.read()
    model = pickle.loads(model_pickled)

bot = telebot.TeleBot('1223773003:AAFG7p47tfm27MG3NpIWDiwOarFB1kheEmY')
root = os.getcwd() + "/dataset/"

def log(text):
    time_stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    print(time_stamp + " " + text)

def save_ogg(ogg_data, ogg_path):
    with open(ogg_path, "wb") as file:
        file.write(ogg_data)


def convert_ogg_wav(ogg_path, dst_path):
    rate = 48000
    cmd = f"ffmpeg -i {ogg_path} -ar {rate} {dst_path} -y -loglevel panic"
    log(cmd)
    with subprocess.Popen(cmd.split()) as p:
        try:
            p.wait(timeout=2)
        except:
            p.kill()
            p.wait()
            return "timeout"


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    user = message.from_user.id
    text = message.text
    log(f"User ({user}): {text}")

#    users_tasks[user] = generate_task()
    bot.send_message(user,
        "Пожалуйста назовите любые 5 цифры с паузами, я попробую их угадать")

@bot.message_handler(content_types=['voice'])
def get_voice_messages(message):
    user = message.from_user.id
    voice = message.voice
#    if user not in users_tasks:
#        bot.send_message(user, "/start")
    log(f"User ({user}): voice")

    tele_file = bot.get_file(voice.file_id)
    ogg_data = bot.download_file(tele_file.file_path)
#    file_name = users_tasks[user].replace(" ", "_")
    ogg_path = root + "inferense/ogg/unk.ogg"
    wav_path = root + "inferense/wav/unk.wav"
    save_ogg(ogg_data, ogg_path)
    convert_ogg_wav(ogg_path, wav_path)
    wav_path = "dataset/inferense/wav/unk.wav"
#    os.execlp("python3", "python3", "split_by_vad.py", wav_path, "0.1", "0.01", "dataset/inference/unk")
    ret = subprocess.call(["python3", "split_by_vad.py", wav_path, "0.1", "0.01", "dataset/inferense/unk"])
    if ret != 0:
        bot.send_message(user, "Вы записали не качественное аудио. Пожалуйста, попробуйте еще раз!")
    else:
        answer = predict()
        bot.send_message(user, f"Спасибо, цифры которые вы назвали:\n{answer[0]} {answer[1]} {answer[2]} {answer[3]} {answer[4]} ")

def predict():
    features= []
    for i in range(5):
        file_path = f"dataset/inferense/unk/unk{i}.wav"
        sample_rate, audio = read(file_path)
        max_duration_sec = 0.6
        max_duration = int(max_duration_sec * sample_rate + 1e-6)
        if len(audio) < max_duration:
            audio = np.pad(audio, (0, max_duration - len(audio)), constant_values=0)
        feature = librosa.feature.melspectrogram(audio.astype(float), sample_rate, n_mels=16, fmax=1000)
#        features_flatten[i] = feature.reshape(-1)
        features.append(feature)
    features_arr = np.array(features)
    d2_features_arr = features_arr.reshape((5, 16*57))
    answer = model.predict(d2_features_arr)
    return answer

if __name__ == "__main__":
    while True:
        try:
            bot.polling(none_stop=True, interval=0)
        except KeyboardInterrupt as e:
            exit(0)
        except Exception as e:
            print(e)
        time.sleep(5)
        print("LOOP")
        print("pid", os.getpid())

#    bot.polling(none_stop=True, interval=0)
