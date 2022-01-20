import os
import sys

from config import *
from util import *

import numpy as np
from keras.models import model_from_json
from tqdm import tqdm
from librosa.util import find_files
from pydub import AudioSegment

if __name__ == '__main__':
    # load test audio and convert to mag/phase
    mix_wav_mag, mix_wav_phase = load_audio(sys.argv[1])

    if not os.path.exists('./_temp'):
        os.makedirs('./_temp')

    # load saved model
    file = open('./models/colab-diff3-mse/model.json', 'r')
    model_json = file.read()
    file.close()
    model = model_from_json(model_json)
    # load weights
    model.load_weights('./models/colab-diff3-mse/model.h5')

    # predict and write into file
    print("Predicting...")
    for index in tqdm(range(0, mix_wav_mag.shape[1], patch_size)):
            if(index+patch_size > mix_wav_mag.shape[1]):
                offset = (index+patch_size) - mix_wav_mag.shape[1] 
                w = ((0,0), (0,offset))
                mix_wav_mag = np.pad(mix_wav_mag, w, 'constant')
                mix_wav_phase = np.pad(mix_wav_phase, w, 'constant')

            mix_mag = mix_wav_mag[:, index:index+patch_size]
            mix_phase = mix_wav_phase[:, index:index+patch_size]
            pred_mag = model.predict(mix_mag[1:].reshape(1, 512, patch_size, 1), batch_size=BATCH)
            pred_mag = np.vstack((np.zeros((patch_size)), pred_mag.reshape(512, patch_size)))
            save_audio("./_temp/pred" + str(index//patch_size)+".wav", pred_mag, mix_phase)

    filenames = find_files("./_temp",ext="wav")
    combined = AudioSegment.empty()
    for filename in filenames:
        audio = AudioSegment.from_wav(filename)
        combined += audio
        os.remove(filename)
    try:
        os.rmdir("./_temp")
    except PermissionError:
        pass
    combined = combined.set_channels(2)
    combined.export("./output.wav", format="wav")
