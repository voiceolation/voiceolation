import librosa
from librosa.util import find_files
from librosa import load
import os
import numpy as np
from config import *
from tqdm import tqdm
from scipy.signal import butter, filtfilt

db_train = "./musdb/train"

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def process(): 
    parent = os.path.dirname(os.path.dirname(db_train))
    db = os.listdir(db_train)
    for song_dir in tqdm(db):
        song_path = db_train + '/' + song_dir
        tracks = find_files(song_path,ext="wav")

        # tracks[0] --> accompaniment (instrumental)
        # tracks[4] --> mixture
        # tracks[6] --> vocals
        #inst,_ = load(tracks[0], sr=None)
        mix,_ = load(tracks[4], sr=None)
        vocal,_ = load(tracks[6], sr=None)

        #44100 is sample rate of musdb18 dataset
        mix = librosa.core.resample(mix,44100,SR)
        vocal = librosa.core.resample(vocal,44100,SR)
        #inst = librosa.core.resample(inst,44100,SR)

        S_mix = butter_lowpass_filter(mix, window_size, SR, order=5)
        S_vocal = butter_lowpass_filter(vocal, window_size, SR, order=5)

        S_mix = librosa.stft(S_mix,n_fft=window_size,hop_length=hop_length).astype(np.float32)
        #S_inst = np.abs(librosa.stft(inst,n_fft=window_size,hop_length=hop_length)).astype(np.float32)
        S_vocal = librosa.stft(S_vocal,n_fft=window_size,hop_length=hop_length).astype(np.float32)

        Y_mix = np.abs(S_mix) ** 2
        #Y_inst
        Y_vocal = np.abs(S_vocal) ** 2

        Y_log_mix = librosa.power_to_db(Y_mix)
        Y_log_vocal = librosa.power_to_db(Y_vocal)
        
        norm1 = Y_log_mix.max()
        norm2 = Y_log_vocal.max()
        Y_log_mix /= norm1
        #S_inst /= norm
        Y_log_vocal /= norm2
        
        spec_dir = parent + '/spectrogram/' + song_dir
        np.savez(spec_dir,mix=S_mix, vocal=S_vocal)

if __name__ == '__main__':
    process()
