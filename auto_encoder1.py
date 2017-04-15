
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

ENC_DIM = 32
NUM_EPOCHS = 100
SRC_DIM = (28,28)
SRC_SIZE = SRC_DIM[0]*SRC_DIM[1]
NUM_IMG = 10
OFFSET_IMG  = 25

# this is our input placeholder
input_img = Input(shape=(SRC_SIZE,))
# "encoded" is the encoded representation of the input
encoded = Dense(ENC_DIM, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(SRC_SIZE, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(ENC_DIM,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

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



plt.figure(figsize=(20,4))
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




