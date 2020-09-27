from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
import numpy as np
from keras.models import model_from_json
import matplotlib
import h5py
import json
import io
import glob
from PIL import Image

MODEL_NAME_AUTO = 'autoencoder3'
MODEL_NAME_SR1 = 'srcnn1'
MODEL_NAME = MODEL_NAME_SR1
NUM_EPOCHS = 200
SRC_DIM = (24,24)
SRC_SIZE = SRC_DIM[0]*SRC_DIM[1]
NUM_IMG = 5
OFFSET_IMG  = 0
LAYER1_SIZE = 5
LAYER2_SIZE = 3

DEBUG_MODE = 0
TRAIN_MODEL = 1
BATCH_SIZE = 16
STEP_FOR_PREDICTION = NUM_EPOCHS-1

PLOT_WEIGHTS = 1
STORE_WEIGHTS = 0
PLOT_RESPONSE = 1
ANIMATE_KERNELS = 0

# if ANIMATE_KERNELS:
#     matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def prep_dataset():
	#print('--- Input Training DataSet ---')
	filelist = glob.glob('./data/fire_LR_x4/train/*_cropped.png')
	#print(filelist)
	x_train = np.array([np.array(Image.open(fname)) for fname in filelist])
	#print(x_train.shape)
	#print('--- Input Validation DataSet ---')
	filelist = glob.glob('./data/fire_LR_x4/valid/*_cropped.png')
	#print(filelist)
	x_val = np.array([np.array(Image.open(fname)) for fname in filelist])
	#print(x_val.shape)
	#print('--- Output Training DataSet ---')
	filelist = glob.glob('./data/fire_HR/train/*_cropped.png')
	#print(filelist)
	y_train = np.array([np.array(Image.open(fname)) for fname in filelist])
	y_train_mono = y_train[:,:,:,0]
	#print(y_train_mono.shape)
	#print('--- Output Validation DataSet ---')
	filelist = glob.glob('./data/fire_HR/valid/*_cropped.png')
	#print(filelist)
	y_val = np.array([np.array(Image.open(fname)) for fname in filelist])
	y_val_mono = y_val[:,:,:,0]
	#print(y_val_mono.shape)
	return x_train, x_val, y_train_mono, y_val_mono

def build_autoencoder():
	# build network model
	input_img = Input(shape=(SRC_DIM[0], SRC_DIM[1], 3))
	x = Conv2D(16, (LAYER1_SIZE,LAYER1_SIZE), activation='relu', padding='same')(input_img)
	x = MaxPooling2D((2,2), padding='same')(x)
	x = Conv2D(8, (LAYER2_SIZE,LAYER2_SIZE), activation='relu', padding='same')(x)
	x = MaxPooling2D((2,2), padding='same')(x)
	x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2,2), padding='same')(x)
	x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
	x = UpSampling2D((2,2))(x)
	x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
	x = UpSampling2D((2,2))(x)
	x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
	x = UpSampling2D((8,8))(x)
	decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	plot_model(autoencoder, to_file='./model/'+MODEL_NAME+'_graph.png', show_shapes=True, show_layer_names=True)
	print(autoencoder.summary())
	model_json = autoencoder.to_json()

	return autoencoder, model_json

def build_srcnn():
	# build network model
	input_img = Input(shape=(SRC_DIM[0], SRC_DIM[1], 3))
	x = Conv2D(16, (LAYER1_SIZE,LAYER1_SIZE), activation='relu', padding='same')(input_img)
	#x = BatchNormalization()(x)
	x = UpSampling2D((2,2), interpolation='bilinear')(x)
	x = Conv2D(8, (LAYER2_SIZE,LAYER2_SIZE), activation='relu', padding='same')(x)
	x = Conv2D(1, (3,3), activation='relu', padding='same')(x)
	sr = UpSampling2D((2,2), interpolation='bilinear')(x)

	srcnn = Model(input_img, sr)
	#opt = SGD(lr=0.01, momentum=0.9)
	#srcnn.compile(optimizer=opt, loss='mean_squared_error')
	srcnn.compile(optimizer='adam', loss='mean_squared_error')

	plot_model(srcnn, to_file='./model/'+MODEL_NAME+'_graph.png', show_shapes=True, show_layer_names=True)
	print(srcnn.summary())
	model_json = srcnn.to_json()

	return srcnn, model_json


if __name__ == '__main__':

	# autoencoder, model_json = build_autoencoder()
	x_train, x_val, y_train_mono, y_val_mono = prep_dataset()

	model, model_json = build_srcnn()

	with open('./model/'+MODEL_NAME+'_model.json', 'w') as json_file:
		json.dump(model_json, json_file, indent=4, sort_keys=True)

	x_train = x_train.astype('float32') / 255.
	x_val = x_val.astype('float32') / 255.
	y_train_mono = y_train_mono.astype('float32') / 255.
	y_val_mono = y_val_mono.astype('float32') / 255.
	print(x_train.shape)
	print(x_val.shape)
	print(y_train_mono.shape)
	print(y_val_mono.shape)

	# x_train = np.reshape(x_train, (len(x_train), SRC_DIM[0], SRC_DIM[1], 3))  # adapt this if using `channels_first` image data format
	# x_val = np.reshape(x_val, (len(x_val), SRC_DIM[0], SRC_DIM[1], 3))  # adapt this if using `channels_first` image data format
	# y_train_mono = np.reshape(y_train_mono, (len(y_train_mono), SRC_DIM[0]*4, SRC_DIM[1]*4, 1))  # adapt this if using `channels_first` image data format
	# y_val_mono = np.reshape(y_val_mono, (len(x_test), SRC_DIM[0]*4, SRC_DIM[1]*4, 1))  # adapt this if using `channels_first` image data format


	if TRAIN_MODEL:
		# Need to remove, adding for debug
		if DEBUG_MODE:
			x_train = x_train[:100,:,:]
			x_test = x_test[:100,:,:]

		model_checkpoint = ModelCheckpoint('./model/'+MODEL_NAME+'_model_{epoch:03d}.hdf5')
		csv_log = CSVLogger('./model/'+MODEL_NAME+'_training_log.csv', separator=',', append=False)
		model.fit(x_train, y_train_mono, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_data=(x_val, y_val_mono), callbacks=[model_checkpoint, csv_log, TensorBoard(log_dir='./model/logs')])
	else:
		step = STEP_FOR_PREDICTION
		model_file = './model/'+MODEL_NAME+'_model_'+str(step).zfill(3)+'.hdf5'
		print(model_file)
		model.load_weights(model_file)

	decoded_imgs = model.predict(x_val)

	# noise_images = np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 

	if PLOT_RESPONSE:
		fig = plt.figure(figsize=(20,12))
		for n in range(NUM_IMG):
			print(np.max(x_val[OFFSET_IMG+n][:,:,0]))
			# if n ==0:
			# 	print('Fire :\n')
			# 	print(x_val[OFFSET_IMG+n][:,:,0])
			# 	print('LC :\n')
			# 	print(x_val[OFFSET_IMG+n][:,:,1])
			# 	print('Temp :\n')
			# 	print(x_val[OFFSET_IMG+n][:,:,0])
			ax = plt.subplot(4, NUM_IMG, n+1)
			plt.imshow(x_val[OFFSET_IMG+n])
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax = plt.subplot(4, NUM_IMG, NUM_IMG+n+1)
			plt.imshow(x_val[OFFSET_IMG+n][:,:,0])
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax = plt.subplot(4, NUM_IMG, 2*NUM_IMG+n+1)
			plt.imshow(y_val_mono[OFFSET_IMG+n]) #.reshape(SRC_DIM[0], SRC_DIM[1]))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax = plt.subplot(4, NUM_IMG, 3*NUM_IMG+n+1)
			plt.imshow(decoded_imgs[OFFSET_IMG+n]) #.reshape(SRC_DIM[0], SRC_DIM[1]))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		fig.suptitle('ROW1: Input 3 channel image; ROW2: Input Fire; ROW3: Output Fire 4xDownscale; ROW4: Predictor 4xDownscale', fontsize=20)
		plt.savefig('./model/'+MODEL_NAME+'_response.png')
		plt.show()

	if STORE_WEIGHTS:
		for step in range(1,NUM_EPOCHS):
			model_file = './model/'+MODEL_NAME+'_model_'+str(step).zfill(3)+'.hdf5'
			print(model_file)
			model.load_weights(model_file)

			w1=model.get_weights()[0]
			for n in range(w1.shape[3]):
				ax = plt.subplot(4, 4, n+1)
				plt.imshow(w1[:,:,0,n].reshape(LAYER1_SIZE,LAYER1_SIZE))
				plt.gray()
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)

			title_string = 'AutoEncoder Layer 1 - Step: ' + str(step)
			plt.suptitle(title_string)
			plt.savefig('./model/'+MODEL_NAME+'_weights-'+str(step).zfill(3)+'.png')


		# if PLOT_WEIGHTS:
		# 	print('Plotting weights')

		# 	w1=autoencoder.get_weights()[0]
		# 	plt.figure(figsize=(10,10))
		# 	for n in range(w1.shape[3]):
		# 		ax = plt.subplot(4, 4, n+1)
		# 		plt.imshow(w1[:,:,0,n].reshape(LAYER1_SIZE,LAYER1_SIZE))
		# 		plt.gray()
		# 		ax.get_xaxis().set_visible(False)
		# 		ax.get_yaxis().set_visible(False)
		# 	w2=autoencoder.get_weights()[2]
		# 	plt.figure(figsize=(10,10))
		# 	for n in range(w2.shape[3]):
		# 		ax = plt.subplot(2, 4, n+1)
		# 		plt.imshow(np.mean(w2,axis=2)[:,:,n])
		# 		plt.gray()
		# 		ax.get_xaxis().set_visible(False)
		# 		ax.get_yaxis().set_visible(False)
		# 	plt.savefig('./model/'+MODEL_NAME+'_weights.png')
			# plt.show()



		# if ANIMATE_KERNELS:

		# 	print('Making animation of kernels...')

		# 	steps = range(0, NUM_EPOCHS)
		# 	fig = plt.figure()

		# 	ani = animation.FuncAnimation(fig, animate, steps)
		# 	gif_file = './model/autoencoder2_kernels_layer1.gif'
		# 	ani.save(gif_file, writer='imagemagick', fps=1)
