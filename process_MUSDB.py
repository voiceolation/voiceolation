import librosa
from librosa.util import find_files
from librosa import load
import os
import numpy as np
from config import *
from tqdm import tqdm 

db_train = "./musdb/train"
def process(): 
    parent = os.path.dirname(os.path.dirname(db_train))
    db = os.listdir(db_train)
    for song_dir in tqdm(db):
        song_path = db_train + '/' + song_dir
        tracks = find_files(song_path,ext="wav")

        # tracks[0] --> accompaniment (instrumental)
        # tracks[4] --> mixture
        # tracks[6] --> vocals
        inst,_ = load(tracks[0], sr=None)
        mix,_ = load(tracks[4], sr=None)
        vocal,_ = load(tracks[6], sr=None)

        #44100 is sample rate of musdb18 dataset
        mix = librosa.core.resample(mix,44100,SR)
        vocal = librosa.core.resample(vocal,44100,SR)
        inst = librosa.core.resample(inst,44100,SR)

        S_mix = np.abs(librosa.stft(mix,n_fft=window_size,hop_length=hop_length)).astype(np.float32)
        S_inst = np.abs(librosa.stft(inst,n_fft=window_size,hop_length=hop_length)).astype(np.float32)
        S_vocal = np.abs(librosa.stft(vocal,n_fft=window_size,hop_length=hop_length)).astype(np.float32)
        
        norm = S_mix.max()
        S_mix /= norm
        S_inst /= norm
        S_vocal /= norm
        
        spec_dir = parent + '/spectrogram/' + song_dir
        np.savez(spec_dir,mix=S_mix,inst=S_inst ,vocal=S_vocal)

if __name__ == '__main__':
    process()
