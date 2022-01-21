import tensorflow as tf
from model import u_net
from util import *
from config import *
from keras.models import model_from_json

def train(path, target):
	diff = 2
	for index in range(0, 100, diff):
		print("Spectrograms are loading..") 
		x_list, y_list = load_spectrogram(path, target,index, index+diff)
		print("Magnitudes of x are calculating..")
		x_mag,x_phase = magphase_list(x_list)
		print("Magnitudes of y are calculating..")
		y_mag,_ = magphase_list(y_list)
		with tf.device('/device:GPU:0'):
			if index == 0:	
				model = u_net()
			else:
			#load json and create model
				file = open('./models/pretrain-{:0>2d}.json'.format((index//diff) -1), 'r')
				model_json = file.read()
				file.close()
				model = model_from_json(model_json)
				# load weights
				model.load_weights('./models/pretrain-{:0>2d}.h5'.format((index//diff) -1))
			#model.summary()
			model.compile(optimizer='adam', loss="mean_squared_error", metrics=['accuracy'])
			for e in range(EPOCH):
				# Random sampling for training
				x,y = sampling(x_mag,y_mag)
				model.fit(x, y, batch_size=BATCH, verbose=1, validation_split=0.01)
				json_file = model.to_json()
				with open('./models/pretrain-{:0>2d}.json'.format(index//diff), "w+") as file:
					file.write(json_file)
				# serialize weights to HDF5
				model.save_weights('./models/pretrain-{:0>2d}.h5'.format(index//diff))
if __name__ == '__main__':
	train("./spectrogram", "vocal")
	   
	print("Training Complete!")
