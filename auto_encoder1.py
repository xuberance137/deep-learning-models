
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

ENC_DIM = 32
NUM_EPOCHS = 50
SRC_DIM = (28,28)
SRC_SIZE = SRC_DIM[0]*SRC_DIM[1]

input_img = Input(shape=(SRC_SIZE,))
encoded = Dense(ENC_DIM, activation='relu')(input_img)
decoded = Dense(SRC_SIZE, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

encoded_input = Input(shape=(ENC_DIM,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32')/255.0
x_test = x_train.astype('float32')/255.0
x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))

autoencoder.fit(x_train, x_train, epochs=NUM_EPOCHS, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

NUM_IMG = 10

plt.figure(figsize=(20,4))
for n in range(NUM_IMG):
	ax = plt.subplot(2, NUM_IMG, n+1)
	plt.imshow(x_test[n].reshape(SRC_DIM[0], SRC_DIM[1]))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(2, NUM_IMG, NUM_IMG+n+1)
	plt.imshow(decoded_imgs[n].reshape(SRC_DIM[0], SRC_DIM[1]))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()




