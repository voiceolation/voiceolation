import numpy as np
from librosa.core import istft, load, stft, magphase
import keras as keras
from librosa.util import find_files
from config import *
from util import * 
from keras.models import model_from_json
import museval 
from tqdm import tqdm
from librosa.util import find_files
from pydub import AudioSegment


if __name__ == '__main__':
	if not os.path.exists('./predict'):
		os.makedirs('./predict')
	if not os.path.exists('./predict/target'):
		os.makedirs('./predict/target')
	if not os.path.exists('./predict/pred'):
		os.makedirs('./predict/pred')
	if not os.path.exists('./predict/target/_temp'):
		os.makedirs('./predict/target/_temp')
	if not os.path.exists('./predict/pred/_temp'):
		os.makedirs('./predict/pred/_temp')

	# load saved model
	#model = keras.models.load_model('../models/vocal_20_test_model.h5')
	file = open("./models/colab-diff3-mse/model.json", 'r')
	model_json = file.read()
	file.close()
	model = model_from_json(model_json)
	# load weights
	model.load_weights('./models/colab-diff3-mse/model.h5')

	# load test audio and convert to mag/phase
	filelist = find_files("./spectrogram-test", ext="npz")
	for file in tqdm(filelist):
		data = np.load(file)
		mix_spec = (data['mix'])
		vocal_spec = (data['vocal'])
		mix_mag, mix_phase = librosa.magphase(mix_spec)
		vocal_mag, vocal_phase = librosa.magphase(vocal_spec)
		filename = os.path.basename(file)
		filename = (filename[:-4])
		for index in range(0, mix_mag.shape[1], patch_size):
			if(index+patch_size > mix_mag.shape[1]):
				offset = (index+patch_size) - mix_mag.shape[1] 
				w = ((0,0), (0,offset))
				mix_mag = np.pad(mix_mag, w, 'constant')
				mix_phase = np.pad(mix_phase, w, 'constant')
				vocal_mag = np.pad(vocal_mag, w, 'constant')
				vocal_phase = np.pad(vocal_phase, w, 'constant')

			partial_mix_mag = mix_mag[:, index:index+patch_size]
			partial_mix_phase = mix_phase[:, index:index+patch_size]
			partial_vocal_mag = vocal_mag[:, index:index+patch_size]
			partial_vocal_phase = vocal_phase[:, index:index+patch_size]
			x = partial_mix_mag[1:].reshape(1, 512, patch_size, 1)
			y = model.predict(x, batch_size=BATCH)
			pred_mag = np.vstack((np.zeros((patch_size)), y.reshape(512, patch_size)))
			save_audio("./predict/target/_temp/" + filename + str(index//patch_size)+ ".wav",  partial_vocal_mag,  partial_vocal_phase)
			save_audio("./predict/pred/_temp/" + filename + str(index//patch_size)+ ".wav", pred_mag, partial_mix_phase)
		filenames = find_files("./predict/target/_temp",ext="wav")
		combined = AudioSegment.empty()
		for f in filenames:
			audio = AudioSegment.from_wav(f)
			combined += audio
			os.remove(f) 
		combined = combined.set_channels(2)
		combined.export("./predict/target/" + filename + ".wav", format="wav")

		filenames = find_files("./predict/pred/_temp",ext="wav")
		combined = AudioSegment.empty()
		for f in filenames:
			audio = AudioSegment.from_wav(f)
			combined += audio
			os.remove(f)
		combined = combined.set_channels(2)
		combined.export("./predict/pred/" + filename + ".wav", format="wav")

	try:
		os.rmdir('./predict/pred/_temp')
		os.rmdir('./predict/target/_temp')
	except PermissionError:
		pass 

	print("Evaluating...")
	eval = museval.eval_dir("./predict/target/", "./predict/pred/", output_dir=None, mode='v4', win=1.0, hop=1.0)
	print(eval)
