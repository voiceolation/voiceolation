import librosa
from librosa.util import find_files
from librosa import load
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display

import os, re 
import numpy as np
from config import *

from tqdm import tqdm 

def plot_spectrogram(file_path):
    y, sr = load(file_path,sr=SR)
    y = librosa.stft(y,n_fft=window_size,hop_length=hop_length)
    Ydb = librosa.amplitude_to_db(abs(y))
    librosa.display.specshow(Ydb, 
                             sr=SR, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis="linear")
    plt.colorbar(format="%+2.f")
    plt.show()

def load_audio(file_path):
    y, sr = load(file_path,sr=SR)
    stft = librosa.stft(y,n_fft=window_size,hop_length=hop_length)
    mag, phase = librosa.magphase(stft)
    return mag.astype(np.float32), phase

def save_audio(file_path, mag, phase):
    y = librosa.istft(mag*phase,win_length=window_size,hop_length=hop_length)
    sf.write(file_path,y,SR)

def load_spectrogram(path_of_spectrograms, target, start, end):
    filelist = find_files(path_of_spectrograms, ext="npz")
    x = []
    y = []
    for index, file in enumerate(filelist):
        if index < start:
            continue
        if index == end:
            break
        print(index)
        data = np.load(file)
        x.append(data['mix'])
        if target == "vocal":
            y.append(data['vocal'])
        else:
            y.append(data['inst'])
        index = index + 1
        
    return x, y

def magphase_list(list_of_spectrogram):
    magnitude_list = []
    phase_list = []
    for spec in tqdm(list_of_spectrogram):
        mag, phase = librosa.magphase(spec)
        magnitude_list.append(mag)
        phase_list.append(phase)
    return magnitude_list, phase_list

def sampling(x_mag,y_mag):
    x = []
    y = []
    for mix, target in zip(x_mag,y_mag) :
        starts = np.random.randint(0, mix.shape[1] - patch_size, (mix.shape[1] - patch_size) // SAMPLING_STRIDE)
        for start in starts:
            end = start + patch_size
            x.append(mix[1:, start:end, np.newaxis])
            y.append(target[1:, start:end, np.newaxis])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)