
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

ENC_DIM = 32
NUM_EPOCHS = 20
SRC_DIM = (28,28)
SRC_SIZE = SRC_DIM[0]*SRC_DIM[1]
NUM_IMG = 10
OFFSET_IMG  = 25

input_img = Input(shape=(SRC_SIZE,))
#encoded = Dense(ENC_DIM, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
encoded = Dense(ENC_DIM, activation='relu')(input_img)
decoded = Dense(SRC_SIZE, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(ENC_DIM,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

encodedDeep = Dense(ENC_DIM*4, activation='relu')(input_img)
encodedDeep = Dense(ENC_DIM*2, activation='relu')(encodedDeep)
encodedDeep = Dense(ENC_DIM, activation='sigmoid')(encodedDeep)
decodedDeep = Dense(ENC_DIM*2, activation='relu')(encodedDeep)
decodedDeep = Dense(ENC_DIM*4, activation='relu')(decodedDeep)
decodedDeep = Dense(SRC_SIZE, activation='sigmoid')(decodedDeep)
autoencoderDeep = Model(input_img, decodedDeep)
autoencoderDeep.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))
print x_train.shape
print x_test.shape
autoencoder.fit(x_train, x_train, epochs=NUM_EPOCHS, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

autoencoderDeep.fit(x_train, x_train, epochs=NUM_EPOCHS, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
decoded_imgsDeep = autoencoderDeep.predict(x_test)
print decoded_imgsDeep.shape

plt.figure(figsize=(20,6))
for n in range(NUM_IMG):
	ax = plt.subplot(3, NUM_IMG, n+1)
	plt.imshow(x_test[OFFSET_IMG+n].reshape(SRC_DIM[0], SRC_DIM[1]))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	ax = plt.subplot(3, NUM_IMG, NUM_IMG+n+1)
	plt.imshow(decoded_imgs[OFFSET_IMG+n].reshape(SRC_DIM[0], SRC_DIM[1]))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	ax = plt.subplot(3, NUM_IMG, 2*NUM_IMG+n+1)
	plt.imshow(decoded_imgsDeep[OFFSET_IMG+n].reshape(SRC_DIM[0], SRC_DIM[1]))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()




