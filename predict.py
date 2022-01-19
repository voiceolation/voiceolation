import numpy as np
from librosa.core import istft, load, stft, magphase
import keras as keras

from config import *
from util import * 
from keras.models import model_from_json

if __name__ == '__main__':
    # load test audio and convert to mag/phase
    music_path = "./mixture.wav"
    mix_wav_mag, mix_wav_phase = load_audio("./mixture.wav")
    vocal_wav_mag, vocal_wav_phase = load_audio("./vocals.wav")
    START = 60
    END = START + patch_size  # 11 seconds

    mix_wav_mag=mix_wav_mag[:, START:END]
    mix_wav_phase=mix_wav_phase[:, START:END]

    vocal_wav_mag=vocal_wav_mag[:, START:END]
    vocal_wav_phase=vocal_wav_phase[:, START:END]
    # load saved model
    #model = keras.models.load_model('../models/vocal_20_test_model.h5')
    file = open('./models/colab-diff3-mse/model.json', 'r')
    model_json = file.read()
    file.close()
    model = model_from_json(model_json)
        # load weights
    model.load_weights('./models/colab-diff3-mse/model.h5')

    # predict and write into file
    print("Predicting...")
    x=mix_wav_mag[1:].reshape(1, 512, patch_size, 1)
    y=model.predict(x, batch_size=BATCH)

    target_pred_mag = np.vstack((np.zeros((patch_size)), y.reshape(512, patch_size)))
    save_audio("./pred.wav",target_pred_mag,mix_wav_phase)
    save_audio("./target.wav",vocal_wav_mag,vocal_wav_phase)

    plot_spectrogram("./pred.wav")
    plot_spectrogram("./target.wav")