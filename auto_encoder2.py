from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
import numpy as np
from keras.models import model_from_json
import matplotlib
import h5py
import json
import io

NUM_EPOCHS = 20
SRC_DIM = (28,28)
SRC_SIZE = SRC_DIM[0]*SRC_DIM[1]
NUM_IMG = 10
OFFSET_IMG  = 25
LAYER1_SIZE = 5
LAYER2_SIZE = 3
PLOT_WEIGHTS = 1
PLOT_RESPONSE = 0
ANIMATE_KERNELS = 1
DEBUG_MODE = 0

if ANIMATE_KERNELS:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

input_img = Input(shape=(SRC_DIM[0], SRC_DIM[1], 1))
x = Conv2D(16, (LAYER1_SIZE,LAYER1_SIZE), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (LAYER2_SIZE,LAYER2_SIZE), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (LAYER2_SIZE,LAYER2_SIZE), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (LAYER2_SIZE,LAYER2_SIZE), activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

model_json = autoencoder.to_json()
with open('./model/autoencoder2_model.json', 'w') as json_file:
	json.dump(model_json, json_file, indent=4, sort_keys=True)

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

# Need to remove, adding for debug
if DEBUG_MODE:
	x_train = x_train[:1000,:,:]
	x_test = x_test[:100,:,:]

model_checkpoint = ModelCheckpoint('./model/autoencoder2_model_{epoch:03d}.hdf5')
csv_log = CSVLogger('./model/autoencoder2_training_log.csv', separator=',', append=False)
autoencoder.fit(x_train, x_train, epochs=NUM_EPOCHS, batch_size=128, shuffle=True, validation_data=(x_test, x_test), callbacks=[model_checkpoint, csv_log, TensorBoard(log_dir='/tmp/autoencoder')])

decoded_imgs = autoencoder.predict(x_test)

noise_images = np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 

if PLOT_RESPONSE:
	plt.figure(figsize=(20,6))
	for n in range(NUM_IMG):
		ax = plt.subplot(2, NUM_IMG, n+1)
		plt.imshow(x_test[OFFSET_IMG+n].reshape(SRC_DIM[0], SRC_DIM[1]))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax = plt.subplot(2, NUM_IMG, NUM_IMG+n+1)
		plt.imshow(decoded_imgs[OFFSET_IMG+n].reshape(SRC_DIM[0], SRC_DIM[1]))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

if PLOT_WEIGHTS:
	print 'Plotting weights'

	w1=autoencoder.get_weights()[0]
	plt.figure(figsize=(10,10))
	for n in range(w1.shape[3]):
		ax = plt.subplot(4, 4, n+1)
		plt.imshow(w1[:,:,0,n].reshape(LAYER1_SIZE,LAYER1_SIZE))
		plt.gray()
		ax.get_xaxis().set_visible(False)

	w2=autoencoder.get_weights()[2]
	plt.figure(figsize=(10,10))
	for n in range(w2.shape[3]):
		ax = plt.subplot(2, 4, n+1)
		plt.imshow(np.mean(w2,axis=2)[:,:,n])
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

def animate(step):
	plt.clf()
	model_file = './model/autoencoder2_model_'+str(step).zfill(3)+'.hdf5'
	print model_file
	autoencoder.load_weights(model_file)

	w1=autoencoder.get_weights()[0]
	for n in range(w1.shape[3]):
		ax = plt.subplot(4, 4, n+1)
		plt.imshow(w1[:,:,0,n].reshape(LAYER1_SIZE,LAYER1_SIZE))
		plt.gray()
		ax.get_xaxis().set_visible(False)

	title_string = 'AutoEncoder Layer 1 - Step: ' + str(step)
	plt.suptitle(title_string)

if ANIMATE_KERNELS:

	print 'Making animation of kernels...'

	steps = range(0, NUM_EPOCHS)
	fig = plt.figure()

	ani = animation.FuncAnimation(fig, animate, steps)
	gif_file = './model/autoencoder2_kernels_layer1.gif'
	ani.save(gif_file, writer='imagemagick', fps=1)






