from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt

NUM_EPOCHS = 20
SRC_DIM = (28,28)
SRC_SIZE = SRC_DIM[0]*SRC_DIM[1]
NUM_IMG = 10
OFFSET_IMG  = 25
LAYER1_SIZE = 5
LAYER2_SIZE = 3
PLOT_WEIGHTS = 1
PLOT_RESPONSE = 0

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

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

autoencoder.fit(x_train, x_train, epochs=NUM_EPOCHS, batch_size=128, shuffle=True, validation_data=(x_test, x_test), callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

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








