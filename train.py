import tensorflow as tf

from model import u_net
from util import *
from config import *
from keras.models import model_from_json

def train(path, target): 

	print("Spectrograms are loading..") 
	x_list, y_list = load_spectrogram(path, target, 9, 18)
	print("Magnitudes of x are calculating..")
	x_mag,x_phase = magphase_list(x_list)
	print("Magnitudes of y are calculating..")
	y_mag,_ = magphase_list(y_list)

	# model = u_net()
	# model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
	

	# checkpointer = tf.keras.callbacks.ModelCheckpoint('model_spectrogram.h5', verbose=1, save_best_only=True)

	# callbacks = [
	#         tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
	#         tf.keras.callbacks.TensorBoard(log_dir='logs')]

	# x,y = sampling(x_mag,y_mag)
	# model.fit(x,y, validation_split=0.1, batch_size=BATCH, epochs=EPOCH, callbacks=callbacks)
	# model.save('../models/model_spectrogram.h5', overwrite=True)




	#load json and create model
	file = open('./models/pretrain05-1.json', 'r')
	model_json = file.read()
	file.close()
	model = model_from_json(model_json)
	# load weights
	model.load_weights('./models/pretrain05-1.h5')
	model.summary()
	model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
	
	for e in range(EPOCH):
		# Random sampling for training
		x,y = sampling(x_mag,y_mag)
		model.fit(x, y, batch_size=BATCH, verbose=1, validation_split=0.1)
		#model.save('../models/vocal_{:0>2d}.h5'.format(e+1), overwrite=True)
		json_file = model.to_json()
		with open('./models/pretrain{:0>2d}-2.json'.format(e+1), "w+") as file:
		   file.write(json_file)
		# serialize weights to HDF5
		model.save_weights('./models/pretrain{:0>2d}-2.h5'.format(e+1))
		
if __name__ == '__main__':
	train("./spectrogram", "vocal")
	   
	print("Training Complete!")