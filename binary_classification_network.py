from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy

import numpy as np

def vectorize_sequences(sequences, dimensions=10000):
	results = np.zeros((len(sequences), dimensions))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1
	return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(nb_words=10000)

word_index = imdb.get_word_index()
reverser_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 0,1,2 reserved for padding, start of sequence and unknown
for n in range(10):
	decoded_review = ''.join([reverser_word_index.get(i-3, '*')+' ' for i in train_data[n]])
	print decoded_review

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = Sequential()
model.add(Dense(16, activation='relu', input_dim=10000))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(lr=0.001), loss=binary_crossentropy, metrics=[binary_accuracy])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val,y_val))

