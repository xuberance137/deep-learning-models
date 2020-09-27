'''
From 
Learning a Deep Convolutional Network for Image Super-Resolution, ECCV2014
f1 = 9, f3 = 5, n1 = 64 and n2 = 32

Composing Weights Animation
convert -delay 10 -loop 0 ./model/srcnn1_weights-*.png ./model/srcnn1_weights-animated.gif
'''


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
NUM_EPOCHS = 500
CROPPED_IMAGES = False
if CROPPED_IMAGES:
	SRC_DIM = (24,24)
else:
	SRC_DIM = (26, 59)
SRC_SIZE = SRC_DIM[0]*SRC_DIM[1]
DOWNSCALE_FACTOR = 4
NUM_IMG = 5
OFFSET_IMG  = 10
LAYER1_SIZE = 9
LAYER2_SIZE = 5
LAYER3_SIZE = 3

DEBUG_MODE = 0
TRAIN_MODEL = 1
BATCH_SIZE = 16
STEP_FOR_PREDICTION = NUM_EPOCHS-1

PLOT_WEIGHTS = 0
STORE_WEIGHTS = 1
PLOT_RESPONSE = 0
ANIMATE_KERNELS = 0

# if ANIMATE_KERNELS:
#     matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def prep_dataset():
	if CROPPED_IMAGES:
		#print('--- Input Training DataSet ---')
		filelist = glob.glob('./data/fire_LR_x4/train/cropped/*_cropped.png')
		#print(filelist)
		x_train = np.array([np.array(Image.open(fname)) for fname in filelist])
		#print(x_train.shape)
		#print('--- Input Validation DataSet ---')
		filelist = glob.glob('./data/fire_LR_x4/valid/cropped/*_cropped.png')
		#print(filelist)
		x_val = np.array([np.array(Image.open(fname)) for fname in filelist])
		#print(x_val.shape)
		#print('--- Output Training DataSet ---')
		filelist = glob.glob('./data/fire_HR/train/cropped/*_cropped.png')
		#print(filelist)
		y_train = np.array([np.array(Image.open(fname)) for fname in filelist])
		y_train_mono = y_train[:,:,:,0]
		#print(y_train_mono.shape)
		#print('--- Output Validation DataSet ---')
		filelist = glob.glob('./data/fire_HR/valid/cropped/*_cropped.png')
		#print(filelist)
		y_val = np.array([np.array(Image.open(fname)) for fname in filelist])
		y_val_mono = y_val[:,:,:,0]
		#print(y_val_mono.shape)
	else:
		#print('--- Input Training DataSet ---')
		filelist = glob.glob('./data/fire_LR_x4/train/*.png')
		#print(filelist)
		x_train = np.array([np.array(Image.open(fname)) for fname in filelist])
		#print(x_train.shape)
		#print('--- Input Validation DataSet ---')
		filelist = glob.glob('./data/fire_LR_x4/valid/*.png')
		#print(filelist)
		x_val = np.array([np.array(Image.open(fname)) for fname in filelist])
		#print(x_val.shape)
		#print('--- Output Training DataSet ---')
		filelist = glob.glob('./data/fire_HR/train/*.png')
		#print(filelist)
		y_train = np.array([np.array(Image.open(fname)) for fname in filelist])
		y_train_mono = y_train[:,:,:,0]
		#print(y_train_mono.shape)
		#print('--- Output Validation DataSet ---')
		filelist = glob.glob('./data/fire_HR/valid/*.png')
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

# def build_srcnn():
# 	# build network model
# 	input_img = Input(shape=(SRC_DIM[0], SRC_DIM[1], 3))
# 	x = Conv2D(16, (LAYER1_SIZE,LAYER1_SIZE), activation='relu', padding='same')(input_img)
# 	#x = BatchNormalization()(x)
# 	x = UpSampling2D((2,2), interpolation='bilinear')(x)
# 	x = Conv2D(8, (LAYER2_SIZE,LAYER2_SIZE), activation='relu', padding='same')(x)
# 	x = Conv2D(1, (3,3), activation='relu', padding='same')(x)
# 	sr = UpSampling2D((2,2), interpolation='bilinear')(x)
# 	#x = Conv2D(1, (1,1), activation='relu', padding='same')(x)

# 	srcnn = Model(input_img, sr)
# 	#opt = SGD(lr=0.01, momentum=0.9)
# 	#srcnn.compile(optimizer=opt, loss='mean_squared_error')
# 	srcnn.compile(optimizer='adam', loss='mean_squared_error')

# 	plot_model(srcnn, to_file='./model/'+MODEL_NAME+'_graph.png', show_shapes=True, show_layer_names=True)
# 	print(srcnn.summary())
# 	model_json = srcnn.to_json()

# 	return srcnn, model_json

def build_srcnn():
	# build network model
	input_img = Input(shape=(SRC_DIM[0], SRC_DIM[1], 3))
	x = Conv2D(16, (LAYER1_SIZE,LAYER1_SIZE), activation='relu', padding='same')(input_img)
	#x = BatchNormalization()(x)
	x = UpSampling2D((2,2), interpolation='bilinear')(x)
	x = Conv2D(8, (LAYER2_SIZE,LAYER2_SIZE), activation='relu', padding='same')(x)
	x = UpSampling2D((2,2), interpolation='bilinear')(x)
	x = Conv2D(8, (LAYER3_SIZE,LAYER3_SIZE), activation='relu', padding='same')(x)
	sr = Conv2D(1, (3,3), activation='relu', padding='same')(x)

	srcnn = Model(input_img, sr)
	#opt = SGD(lr=0.01, momentum=0.9)
	#srcnn.compile(optimizer=opt, loss='mean_squared_error')
	srcnn.compile(optimizer='adam', loss='mean_squared_error')

	plot_model(srcnn, to_file='./model/'+MODEL_NAME+'_graph.png', show_shapes=True, show_layer_names=True)
	# print(srcnn.summary())
	model_json = srcnn.to_json()

	return srcnn, model_json


def compute_metrics(model):
	years = np.array(range(2017, 2020)) #Right now doesn't include the two files for 2020
	months = np.array(range(1, 13))
	mse_bicubic = []
	mse_srcnn = []
	nmse_bicubic = []
	nmse_srcnn = []
	fireOccurrence_bicubic = []
	fireOccurrence_srcnn = []
	nofireOccurrence_bicubic = []
	nofireOccurrence_srcnn = []
	precision_bicubic = []
	precision_srcnn = []
	recall_bicubic = []
	recall_srcnn = []
	f1_bicubic = []
	f1_srcnn = []
	PREDICTOR_THRESHOLD = 0.0

	for year_idx in years:
		for month_idx in months:
			# print(str(year_idx), str(month_idx), sep =' ')

			lr_img = Image.open('./data/fire_LR_x4/valid/fire_lc_tmean_' + str(year_idx) + '_' + str(month_idx) +'.png')
			lr_img_np = np.array(lr_img).astype('float32') / 255.
			# print("LR image shape: ", lr_img_np.shape)
			hr_img = Image.open('./data/fire_HR/valid/fire_lc_tmean_' + str(year_idx) + '_' + str(month_idx) +'.png')
			hr_img_np = np.array(hr_img).astype('float32') / 255.
			# print("HR image shape: ", hr_img_np.shape)
			exp_lr_img_np = np.expand_dims(lr_img_np, axis=0) 
			# print("LR Model Input image shape: ", exp_lr_img_np.shape)
			sr_img_np = model.predict(np.array(exp_lr_img_np))
			# print("SR image shape: ", sr_img_np.shape)

			# Tried BICUBIC, BILINEAR, LANCZOS
			sr_img_bicubic = lr_img.resize(size=(lr_img.size[0]*DOWNSCALE_FACTOR, lr_img.size[1]*DOWNSCALE_FACTOR), resample=Image.BICUBIC)
			sr_img_bicubic_np = np.array(sr_img_bicubic).astype('float32') / 255.

			## MSE and NMSE
			mse_bicubic.append(np.mean(np.square(sr_img_bicubic_np[:,:,0] - hr_img_np[:,:, 0])))
			nmse_bicubic.append(np.mean(np.square(sr_img_bicubic_np[:,:,0] - hr_img_np[:,:, 0]))/np.mean(np.square(hr_img_np[:,:, 0])))

			mse_srcnn.append(np.mean(np.square(sr_img_np[0, :,:,0] - hr_img_np[:,:, 0])))
			nmse_srcnn.append(np.mean(np.square(sr_img_np[0, :,:,0] - hr_img_np[:,:, 0]))/np.mean(np.square(hr_img_np[:,:, 0])))

			## Fire Occurrence
			fireOccurrence_bicubic.append(np.mean(sr_img_bicubic_np[:,:,0][hr_img_np[:,:, 0] > 0] > 0))
			fireOccurrence_srcnn.append(np.mean(sr_img_np[0, :,:,0][hr_img_np[:,:, 0] > 0] > PREDICTOR_THRESHOLD))

			## NoFire Occurrence
			nofireOccurrence_bicubic.append(np.mean(sr_img_bicubic_np[:,:,0][hr_img_np[:,:, 0] == 0] == 0))
			nofireOccurrence_srcnn.append(np.mean(sr_img_np[0,:,:,0][hr_img_np[:,:, 0] == 0] <= PREDICTOR_THRESHOLD))

			precision_bicubic_image = np.mean(sr_img_bicubic_np[:,:,0][hr_img_np[:,:, 0] > 0] > 0)/((np.mean(sr_img_bicubic_np[:,:,0][hr_img_np[:,:, 0] > 0] > PREDICTOR_THRESHOLD)) + np.mean(sr_img_bicubic_np[:,:,0][hr_img_np[:,:, 0] == 0] > PREDICTOR_THRESHOLD))
			precision_srcnn_image = np.mean(sr_img_np[0,:,:,0][hr_img_np[:,:, 0] > 0] > PREDICTOR_THRESHOLD)/((np.mean(sr_img_np[0,:,:,0][hr_img_np[:,:, 0] > 0] > PREDICTOR_THRESHOLD)) + np.mean(sr_img_np[0,:,:,0][hr_img_np[:,:, 0] == 0] > PREDICTOR_THRESHOLD))
			recall_bicubic_image = np.mean(sr_img_bicubic_np[:,:,0][hr_img_np[:,:, 0] > 0] > 0)/((np.mean(sr_img_bicubic_np[:,:,0][hr_img_np[:,:, 0] > 0] > PREDICTOR_THRESHOLD)) + np.mean(sr_img_bicubic_np[:,:,0][hr_img_np[:,:, 0] > 0] <= PREDICTOR_THRESHOLD))
			recall_srcnn_image = np.mean(sr_img_np[0,:,:,0][hr_img_np[:,:, 0] > 0] > PREDICTOR_THRESHOLD)/((np.mean(sr_img_np[0,:,:,0][hr_img_np[:,:, 0] > 0] > PREDICTOR_THRESHOLD)) + np.mean(sr_img_np[0,:,:,0][hr_img_np[:,:, 0] > 0] <= PREDICTOR_THRESHOLD))

			f1_bicubic_image = 2*precision_bicubic_image*recall_bicubic_image/(precision_bicubic_image+recall_bicubic_image)
			f1_srcnn_image = 2*precision_srcnn_image*recall_srcnn_image/(precision_srcnn_image+recall_srcnn_image)

			precision_bicubic.append(precision_bicubic_image)
			precision_srcnn.append(precision_srcnn_image)
			recall_bicubic.append(recall_bicubic_image)
			recall_srcnn.append(recall_srcnn_image)
			f1_bicubic.append(f1_bicubic_image)
			f1_srcnn.append(f1_srcnn_image)

	print("Bicubic MSE: ", np.mean(mse_bicubic)) # Low = good
	print("FireSR MSE: ", np.mean(mse_srcnn))
	print("Bicubic NMSE: ", np.mean(nmse_bicubic)) # Low = good
	print("FireSR NMSE: ", np.mean(nmse_srcnn))
	print("Bicubic TPR: ", np.mean(fireOccurrence_bicubic))
	print("FireSR TPR: ", np.mean(fireOccurrence_srcnn))
	print("Bicubic TNR: ", np.mean(nofireOccurrence_bicubic))
	print("FireSR TNR: ", np.mean(nofireOccurrence_srcnn))
	print("Bicubic Precision: ", np.mean(precision_bicubic))
	print("FireSR Precision: ", np.mean(precision_srcnn))
	print("Bicubic Recall: ", np.mean(recall_bicubic))
	print("FireSR Recall: ", np.mean(recall_srcnn))
	print("Bicubic F1: ", np.mean(f1_bicubic))
	print("FireSR F1: ", np.mean(f1_srcnn))

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
	# print(x_train.shape)
	# print(x_val.shape)
	# print(y_train_mono.shape)
	# print(y_val_mono.shape)

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
		#print(model_file)
		model.load_weights(model_file)

	decoded_imgs = model.predict(x_val)

	compute_metrics(model)

	# noise_images = np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 

	if PLOT_RESPONSE:
		fig = plt.figure(figsize=(20,12))
		for n in range(NUM_IMG):
			# print(np.max(x_val[OFFSET_IMG+n][:,:,0]))
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

