'''
From 
Learning a Deep Convolutional Network for Image Super-Resolution, ECCV2014
f1 = 9, f3 = 5, n1 = 64 and n2 = 32

Composing Weights Animation
convert -delay 10 -loop 0 ./model/srcnn1_weights-*.png ./model/srcnn1_weights-animated.gif

Renaming AUS data files
for f in *.png; do  echo "Moving $f"; mv "$f"  "$(basename "$f" .png)_AUS.png"; done

'''


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout
from keras.models import Model
from keras import backend as K
from keras import optimizers
from keras.datasets import mnist
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
from keras.layers.merge import concatenate
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
from keras.models import model_from_json
import matplotlib
import h5py
import json
import io
import sys
import glob
from PIL import Image

MODEL_NAME_AUTO = 'autoencoder3'
MODEL_NAME_SR1 = 'srcnn1'
MODEL_NAME_SR_ZI = 'srcnn1_zi'
MODEL_NAME_UNET = 'srunet'

CROPPED_IMAGES = False
if CROPPED_IMAGES:
	SRC_DIM = (24,24)
else:
	SRC_DIM = (26, 59)
	PAD_SRC_DIM = (32, 64)
SRC_SIZE = SRC_DIM[0]*SRC_DIM[1]
DOWNSCALE_FACTOR = 4
NUM_IMG = 5
OFFSET_IMG  = 10
LAYER1_SIZE = 9
LAYER2_SIZE = 5
LAYER3_SIZE = 3

DEBUG_MODE = 0
BATCH_SIZE = 16
PREDICTOR_THRESHOLD = 0.0
MODEL_NAME = MODEL_NAME_SR1 #MODEL_NAME_UNET
BINARY_PREDICTOR = 0
NUM_EPOCHS = 1000
AUS_TEST = 1
TRAIN_MODEL = 0
STEP_FOR_PREDICTION = NUM_EPOCHS-1 #41 #100 #350 #NUM_EPOCHS-1

PLOT_WEIGHTS = 0
STORE_WEIGHTS = 0
PLOT_RESPONSE = 1
ANIMATE_KERNELS = 0

# if ANIMATE_KERNELS:
#     matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.set_printoptions(threshold=sys.maxsize) # remove if you want truncation

import sklearn.metrics as metrics
def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    #mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    #print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

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

	for year_idx in years:
		for month_idx in months:
			# print(str(year_idx), str(month_idx), sep =' ')

			if AUS_TEST:
				lr_img = Image.open('./data/AUS/fire_LR_0.25_x4/fire_lc_tmean_' + str(year_idx) + '_' + str(month_idx) +'_AUS.png')
			else:
				lr_img = Image.open('./data/fire_LR_x4/valid/fire_lc_tmean_' + str(year_idx) + '_' + str(month_idx) +'.png')
				
			lr_img_np = np.array(lr_img).astype('float32') / 255.
			# print("LR image shape: ", lr_img_np.shape)
			if AUS_TEST:
				hr_img = Image.open('./data/AUS/fire_HR_0.25/fire_lc_tmean_' + str(year_idx) + '_' + str(month_idx) +'_AUS.png')
			else:
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

	print("Bicubic MSE: ", np.mean(mse_bicubic), '+/-', np.std(mse_bicubic)) 
	print("FireSR MSE: ", np.mean(mse_srcnn), '+/-', np.std(mse_srcnn))
	print("Bicubic NMSE: ", np.mean(nmse_bicubic), '+/-', np.std(nmse_bicubic))
	print("FireSR NMSE: ", np.mean(nmse_srcnn), '+/-', np.std(nmse_srcnn))
	print("Bicubic TPR: ", np.mean(fireOccurrence_bicubic), '+/-', np.std(fireOccurrence_bicubic))
	print("FireSR TPR: ", np.mean(fireOccurrence_srcnn), '+/-', np.std(fireOccurrence_srcnn))
	print("Bicubic TNR: ", np.mean(nofireOccurrence_bicubic), '+/-', np.std(nofireOccurrence_srcnn))
	print("FireSR TNR: ", np.mean(nofireOccurrence_srcnn), '+/-', np.std(nofireOccurrence_srcnn))
	print("Bicubic Precision: ", np.mean(precision_bicubic), '+/-', np.std(precision_bicubic))
	print("FireSR Precision: ", np.mean(precision_srcnn),'+/-', np.std(precision_srcnn))
	print("Bicubic Recall: ", np.mean(recall_bicubic),'+/-', np.std(recall_bicubic))
	print("FireSR Recall: ", np.mean(recall_srcnn),'+/-', np.std(recall_srcnn))
	print("Bicubic F1: ", np.mean(f1_bicubic),'+/-', np.std(f1_bicubic))
	print("FireSR F1: ", np.mean(f1_srcnn),'+/-', np.std(f1_srcnn))

def build_srcnn_simple():
	# build network model
	input_img = Input(shape=(SRC_DIM[0], SRC_DIM[1], 3))
	x = Conv2D(16, (LAYER1_SIZE,LAYER1_SIZE), activation='relu', padding='same')(input_img)
	#x = BatchNormalization()(x)
	x = UpSampling2D((2,2), interpolation='bilinear')(x)
	x = Conv2D(8, (LAYER2_SIZE,LAYER2_SIZE), activation='relu', padding='same')(x)
	x = Conv2D(1, (3,3), activation='relu', padding='same')(x)
	sr = UpSampling2D((2,2), interpolation='bilinear')(x)
	x = Conv2D(1, (1,1), activation='relu', padding='same')(x)

	srcnn = Model(input_img, sr)
	#opt = SGD(lr=0.01, momentum=0.9)
	#srcnn.compile(optimizer=opt, loss='mean_squared_error')
	srcnn.compile(optimizer='adam', loss='mean_squared_error')

	plot_model(srcnn, to_file='./model/'+MODEL_NAME+'_graph.png', show_shapes=True, show_layer_names=True)
	print(srcnn.summary())
	model_json = srcnn.to_json()

	return srcnn, model_json

def build_srcnn_drift():
	# build network model
	input_img = Input(shape=(SRC_DIM[0], SRC_DIM[1], 3))
	x = Conv2D(16, (LAYER1_SIZE,LAYER1_SIZE), activation='relu', padding='same')(input_img)
	#x = BatchNormalization()(x)
	x = UpSampling2D((2,2), interpolation='bilinear')(x)
	x = Conv2D(8, (LAYER2_SIZE,LAYER2_SIZE), activation='relu', padding='same')(x)
	x = Conv2D(8, (LAYER3_SIZE,LAYER3_SIZE), activation='relu', padding='same')(x)
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

def build_srcnn():
	# build network model
	input_img = Input(shape=(SRC_DIM[0], SRC_DIM[1], 3))
	x = Conv2D(16, (LAYER1_SIZE,LAYER1_SIZE), activation='relu', padding='same')(input_img)
	#x = BatchNormalization()(x)
	x = UpSampling2D((2,2), interpolation='bilinear')(x)
	x = Conv2D(8, (LAYER2_SIZE,LAYER2_SIZE), activation='relu', padding='same')(x)
	x = Conv2D(8, (LAYER3_SIZE,LAYER3_SIZE), activation='relu', padding='same')(x)
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

def build_srcnn_zero_inflated0():
	# build network model
	input_img = Input(shape=(SRC_DIM[0], SRC_DIM[1], 3))
	x = Conv2D(16, (LAYER1_SIZE,LAYER1_SIZE), activation='relu', padding='same')(input_img)
	#x = BatchNormalization()(x)
	x = UpSampling2D((2,2), interpolation='bilinear')(x)
	x = Conv2D(8, (LAYER2_SIZE,LAYER2_SIZE), activation='relu', padding='same')(x)
	x = Conv2D(8, (LAYER3_SIZE,LAYER3_SIZE), activation='relu', padding='same')(x)
	x = Conv2D(1, (3,3), activation='relu', padding='same')(x)
	sr = UpSampling2D((2,2), interpolation='bilinear')(x)

	srcnn = Model(input_img, sr)
	#opt = SGD(lr=0.01, momentum=0.9)
	#srcnn.compile(optimizer=opt, loss='mean_squared_error')
	srcnn.compile(optimizer='adam', loss='binary_crossentropy')

	plot_model(srcnn, to_file='./model/'+MODEL_NAME+'_graph.png', show_shapes=True, show_layer_names=True)
	print(srcnn.summary())
	model_json = srcnn.to_json()

	return srcnn, model_json

def build_srcnn_zero_inflated():

	# build network model
	input_img = Input(shape=(SRC_DIM[0], SRC_DIM[1], 3))
	x = Conv2D(16, (LAYER1_SIZE,LAYER1_SIZE), activation='relu', padding='same')(input_img)
	#x = BatchNormalization()(x)
	x = UpSampling2D((2,2), interpolation='bilinear')(x)
	x = Conv2D(8, (LAYER2_SIZE,LAYER2_SIZE), activation='relu', padding='same')(x)
	x = Conv2D(8, (LAYER3_SIZE,LAYER3_SIZE), activation='relu', padding='same')(x)
	x = Conv2D(1, (1,1), activation='sigmoid', padding='same')(x)
	sr = UpSampling2D((2,2), interpolation='bilinear')(x)

	srcnn = Model(input_img, sr)
	srcnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	plot_model(srcnn, to_file='./model/'+MODEL_NAME+'_graph.png', show_shapes=True, show_layer_names=True)
	print(srcnn.summary())
	model_json = srcnn.to_json()

	return srcnn, model_json

def build_binary_unet_lite_segmentation():
	#input_img = Input(shape=(SRC_DIM[0], SRC_DIM[1], 3))
	input_img = Input(shape=(64, 32, 3))
	conv1 = Conv2D(32, (3,3), activation='elu',padding='same')(input_img)
	conv1 = Dropout(0.2)(conv1)
	conv1 = Conv2D(32, (3,3), activation='elu',padding='same', name='conv_1')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2), name='pool_1')(conv1)
	pool1 = BatchNormalization()(pool1)

	conv2 = Conv2D(64, (3,3), activation='elu',padding='same')(pool1)
	conv2 = Dropout(0.2)(conv2)
	conv2 = Conv2D(64, (3,3), activation='elu',padding='same', name='conv_2')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2), name='pool_2')(conv2)
	pool2 = BatchNormalization()(pool2)

	conv3 = Conv2D(128, (3,3), activation='elu',padding='same')(pool2)
	conv3 = Dropout(0.2)(conv3)
	conv3 = Conv2D(128, (3,3), activation='elu',padding='same', name='conv_3')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2), name='pool_3')(conv3)
	pool3 = BatchNormalization()(pool3)

	conv4 = Conv2D(256, (3,3), activation='elu',padding='same')(pool3)
	conv4 = Dropout(0.2)(conv4)
	conv4 = Conv2D(256, (3,3), activation='elu',padding='same', name='conv_4')(conv4)
	conv4 = BatchNormalization()(conv4)
	# pool4 = MaxPooling2D(pool_size=(2, 2), name='pool_4')(conv4)

	# conv5 = Convolution2D(512, options.filter_width, options.stride, activation='elu',border_mode='same')(pool4)
	# conv5 = Dropout(0.2)(conv5)
	# conv5 = Convolution2D(512, options.filter_width, options.stride, activation='elu',border_mode='same', name='conv_5')(conv5)

	# up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
	# conv6 = Convolution2D(256, options.filter_width, options.stride, activation='elu',border_mode='same')(up6)
	# conv6 = Dropout(0.2)(conv6)
	# conv6 = Convolution2D(256, options.filter_width, options.stride, activation='elu',border_mode='same', name='conv_6')(conv6)

	up7 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=3)

	conv7 = Conv2D(128, (3,3), activation='elu',padding='same')(up7)
	conv7 = Dropout(0.2)(conv7)
	conv7 = Conv2D(128, (3,3), activation='elu',padding='same', name='conv_7')(conv7)
	conv7 = BatchNormalization()(conv7)

	up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3,3), activation='elu',padding='same')(up8)
	conv8 = Dropout(0.2)(conv8)
	conv8 = Conv2D(64, (3,3), activation='elu',padding='same', name='conv_8')(conv8)
	conv8 = BatchNormalization()(conv8)

	up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3,3), activation='elu',padding='same')(up9)
	conv9 = Dropout(0.2)(conv9)
	conv9 = Conv2D(32, (3,3), activation='elu',padding='same', name='conv_9')(conv9)
	conv9 = BatchNormalization()(conv9)
	conv10 = UpSampling2D((2,2), interpolation='bilinear',name='up_sample')(conv9)

	conv11 = Conv2D(1, 1, 1, activation='sigmoid', name='sigmoid')(conv10)

	model = Model(input_img, conv11)
	model.summary()
	# model.compile(optimizer=Adam(lr=options.lr, clipvalue=1., clipnorm=1.), loss=dice_coef_loss, metrics=[dice_coef])
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	plot_model(model, to_file='./model/'+MODEL_NAME+'_graph.png', show_shapes=True, show_layer_names=True)
	model_json = model.to_json()
	return model, model_json

def build_binary_unet_lite():  #for 4x upsampling
	#input_img = Input(shape=(SRC_DIM[0], SRC_DIM[1], 3))
	input_img = Input(shape=(64, 32, 3))
	conv1 = Conv2D(8, (3,3), activation='elu',padding='same')(input_img)
	conv1 = Dropout(0.2)(conv1)
	conv1 = Conv2D(8, (3,3), activation='elu',padding='same', name='conv_1')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2), name='pool_1')(conv1)
	pool1 = BatchNormalization()(pool1)

	conv3 = Conv2D(16, (3,3), activation='elu',padding='same')(pool1)
	conv3 = Dropout(0.2)(conv3)
	conv3 = Conv2D(16, (3,3), activation='elu',padding='same', name='conv_3')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2), name='pool_3')(conv3)
	pool3 = BatchNormalization()(pool3)

	conv4 = Conv2D(16, (3,3), activation='elu',padding='same')(pool3)
	conv4 = Dropout(0.2)(conv4)
	conv4 = Conv2D(16, (3,3), activation='elu',padding='same', name='conv_4')(conv4)
	conv4 = BatchNormalization()(conv4)

	up7 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=3)

	conv7 = Conv2D(16, (3,3), activation='elu',padding='same')(up7)
	conv7 = Dropout(0.2)(conv7)
	conv7 = Conv2D(16, (3,3), activation='elu',padding='same', name='conv_7')(conv7)
	conv7 = BatchNormalization()(conv7)

	up9 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv1], axis=3)
	up9 = UpSampling2D((2,2), interpolation='bilinear',name='up_sample_1')(up9)
	conv9 = Conv2D(8, (3,3), activation='elu',padding='same')(up9)
	conv9 = Dropout(0.2)(conv9)
	conv9 = Conv2D(8, (3,3), activation='elu',padding='same', name='conv_9')(conv9)
	conv9 = BatchNormalization()(conv9)

	conv10 = UpSampling2D((2,2), interpolation='bilinear',name='up_sample_2')(conv9)

	conv11 = Conv2D(1, 1, 1, activation='sigmoid', name='sigmoid')(conv10)

	model = Model(input_img, conv11)
	model.summary()
	# model.compile(optimizer=Adam(lr=options.lr, clipvalue=1., clipnorm=1.), loss=dice_coef_loss, metrics=[dice_coef])
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	plot_model(model, to_file='./model/'+MODEL_NAME+'_graph.png', show_shapes=True, show_layer_names=True)
	model_json = model.to_json()
	return model, model_json

if __name__ == '__main__':

	# autoencoder, model_json = build_autoencoder()
	x_train, x_val, y_train_mono, y_val_mono = prep_dataset()

	x_train = x_train.astype('float32') / 255.
	x_val = x_val.astype('float32') / 255.
	y_train_mono = y_train_mono.astype('float32') / 255.
	y_val_mono = y_val_mono.astype('float32') / 255.
	
	output_train_mask_shape = y_train_mono.shape
	output_val_mask_shape = y_val_mono.shape

	if BINARY_PREDICTOR:
		#model, model_json = build_srcnn_zero_inflated()
		model, model_json = build_binary_unet_lite()

		print(output_train_mask_shape)
		print(output_val_mask_shape)
		x_train_padded = np.zeros((x_train.shape[0], PAD_SRC_DIM[0], PAD_SRC_DIM[1], x_train.shape[3]), dtype=float)
		x_train_padded[:,2:2+x_train.shape[1],4:4+x_train.shape[2],:] = x_train[:,:,:,:]
		x_val_padded = np.zeros((x_val.shape[0], PAD_SRC_DIM[0], PAD_SRC_DIM[1], x_val.shape[3]), dtype=float)
		x_val_padded[:,2:2+x_train.shape[1],4:4+x_train.shape[2],:] = x_val[:,:,:,:]

		y_train_binary_mono = np.zeros((output_train_mask_shape[0], DOWNSCALE_FACTOR*PAD_SRC_DIM[0], DOWNSCALE_FACTOR*PAD_SRC_DIM[1]), dtype=float)
		print(y_train_binary_mono.shape)
		print(y_train_mono.shape)
		print(y_train_binary_mono[:,12:12+output_train_mask_shape[1],10:10+output_train_mask_shape[2]].shape)
		y_train_binary_mono[:,12:12+output_train_mask_shape[1],10:10+output_train_mask_shape[2]][y_train_mono[:,:, :] > 0] = 1.0
		y_val_binary_mono = np.zeros((output_val_mask_shape[0], DOWNSCALE_FACTOR*PAD_SRC_DIM[0], DOWNSCALE_FACTOR*PAD_SRC_DIM[1]), dtype=float)
		y_val_binary_mono[:,12:12+output_train_mask_shape[1],10:10+output_train_mask_shape[2]][y_val_mono[:,:, :] > 0] = 1.0

	else:
		model, model_json = build_srcnn()

	with open('./model/'+MODEL_NAME+'_model.json', 'w') as json_file:
		json.dump(model_json, json_file, indent=4, sort_keys=True)


	# print(y_train_mono[0][50:55])
	# print(y_train_binary_mono[0][50:55])
	# # 	# create label encoder
	# le = LabelEncoder()
	# le.fit(y_train_binary_mono)
	# y_train_binary_mono = encoder.transform(y_train_binary_mono)
	# print(y_train_mono[0][50:55])

	# print(x_train.shape)
	# print(x_val.shape)
	# print(y_train_mono.shape)
	# print(y_val_mono.shape)

	# x_train = np.reshape(x_train, (len(x_train), SRC_DIM[0], SRC_DIM[1], 3))  # adapt this if using `channels_first` image data format
	# x_val = np.reshape(x_val, (len(x_val), SRC_DIM[0], SRC_DIM[1], 3))  # adapt this if using `channels_first` image data format
	# y_train_mono = np.reshape(y_train_mono, (len(y_train_mono), SRC_DIM[0]*4, SRC_DIM[1]*4, 1))  # adapt this if using `channels_first` image data format
	# y_val_mono = np.reshape(y_val_mono, (len(x_test), SRC_DIM[0]*4, SRC_DIM[1]*4, 1))  # adapt this if using `channels_first` image data format


	if TRAIN_MODEL:
		model_checkpoint = ModelCheckpoint('./model/'+MODEL_NAME+'_model_{epoch:03d}.hdf5')
		csv_log = CSVLogger('./model/'+MODEL_NAME+'_training_log.csv', separator=',', append=False)
		if BINARY_PREDICTOR:
			model.fit(x_train_padded, y_train_binary_mono, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_data=(x_val_padded, y_val_binary_mono), callbacks=[model_checkpoint, csv_log, TensorBoard(log_dir='./model/logs')])
		else:
			model.fit(x_train, y_train_mono, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_data=(x_val, y_val_mono), callbacks=[model_checkpoint, csv_log, TensorBoard(log_dir='./model/logs')])
	else:
		step = STEP_FOR_PREDICTION
		model_file = './model/'+MODEL_NAME+'_model_'+str(step).zfill(3)+'.hdf5'
		print(model_file)
		model.load_weights(model_file)

	compute_metrics(model)

	# if BINARY_PREDICTOR:
	# 	filelist = glob.glob('./data/fire_HR/train/*.png')
	# 	y_train = np.array([np.array(Image.open(fname)) for fname in filelist])
	# 	sr_temp = y_train[:,:,:,2].astype('float32') / 255.
	# 	sr_lc = y_train[:,:,:,1].astype('float32') / 255.
	# 	predict_train_images = model.predict(x_train)
	# 	sr_fire = np.zeros((predict_train_images.shape[0],predict_train_images.shape[1], predict_train_images.shape[2], predict_train_images.shape[3]), dtype=float)
	# 	sr_fire[:,:,:,:][predict_train_images[:,:,:,:] > PREDICTOR_THRESHOLD] = 1.0
	# 	print(sr_temp.flatten().shape)
	# 	print(sr_lc.flatten().shape)
	# 	print(sr_fire.flatten().shape)
	# 	X = np.vstack((sr_temp.flatten(),sr_lc.flatten(), sr_fire.flatten()))
	# 	Xt = np.transpose(X)
	# 	print(Xt.shape)
	# 	print(Xt[900:1100,:])
	# 	y_train_input = y_train[:,:,:,0].astype('float32') / 255.
	# 	y = y_train_input.flatten() #np.transpose(np.expand_dims(y_train_input.flatten(), axis=0)) 
	# 	print(y.shape)
	# 	reg_model = LinearRegression().fit(Xt, y)

	# 	y_pred = reg_model.predict(Xt)
	# 	regression_results(y, y_pred)

	if AUS_TEST:
		filelist = glob.glob('./data/AUS/fire_LR_0.25_x4/*.png')
		x_val = np.array([np.array(Image.open(fname)) for fname in filelist]).astype('float32') / 255.
		filelist = glob.glob('./data/AUS/fire_HR_0.25/*.png')
		y_val = np.array([np.array(Image.open(fname)) for fname in filelist]).astype('float32') / 255.
		y_val_mono = y_val[:,:,:,0]

	if BINARY_PREDICTOR:
		decoded_imgs_cont = model.predict(x_val_padded)
		decoded_img_mask_shape = decoded_imgs_cont.shape
		print(decoded_img_mask_shape)
		decoded_imgs = np.zeros((decoded_img_mask_shape[0],decoded_img_mask_shape[1], decoded_img_mask_shape[2], decoded_img_mask_shape[3]), dtype=float)
		decoded_imgs[:,:,:,:][decoded_imgs_cont[:,:,:,:] > PREDICTOR_THRESHOLD] = 1.0
	else:
		decoded_imgs = model.predict(x_val)

	
	#print(x_val.shape)
	#print('--- Output Training DataSet ---')

	# noise_images = np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 

	if PLOT_RESPONSE:
		fig = plt.figure(figsize=(16,8))
		for n in range(NUM_IMG):
			print(np.max(x_val[OFFSET_IMG+n][:,:,0]))
			#if n ==0:
				# print('Fire :\n')
				# print([x for x in x_val[OFFSET_IMG+n][:,:,2]])
				# print('LC :\n')
				# print(x_val[OFFSET_IMG+n][:,:,1])
				# print('Temp :\n')
				# print(x_val[OFFSET_IMG+n][:,:,0])
				# print(decoded_imgs[OFFSET_IMG+n])
			ax = plt.subplot(5, NUM_IMG, n+1)
			plt.imshow(x_val[OFFSET_IMG+n][:,:,2])
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax = plt.subplot(5, NUM_IMG, NUM_IMG+n+1)
			plt.imshow(x_val[OFFSET_IMG+n][:,:,1])
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax = plt.subplot(5, NUM_IMG, 2*NUM_IMG+n+1)
			plt.imshow(x_val[OFFSET_IMG+n][:,:,0])
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax = plt.subplot(5, NUM_IMG, 3*NUM_IMG+n+1)
			if BINARY_PREDICTOR:
				plt.imshow(y_val_binary_mono[OFFSET_IMG+n]) #.reshape(SRC_DIM[0], SRC_DIM[1]))
			else:
				plt.imshow(y_val_mono[OFFSET_IMG+n])
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax = plt.subplot(5, NUM_IMG, 4*NUM_IMG+n+1)
			if BINARY_PREDICTOR:
				plt.imshow(decoded_imgs_cont[OFFSET_IMG+n]) #.reshape(SRC_DIM[0], SRC_DIM[1]))
			else:
				plt.imshow(decoded_imgs[OFFSET_IMG+n])
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		fig.suptitle('ROW1: Input Temp; ROW2: Input LC; ROW3: Input Fire; ROW4: Output Fire 4xDownscale; ROW5: Predictor 4xDownscale', fontsize=16)
		plt.savefig('./model/'+MODEL_NAME+'_response.png')
		plt.show()

	if STORE_WEIGHTS:
		for step in range(1,300): #NUM_EPOCHS):
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

			title_string = 'FireSR Layer 1 - Step: ' + str(step)
			plt.suptitle(title_string)
			plt.savefig('./model/'+MODEL_NAME+'_weights-'+str(step).zfill(3)+'.png')

