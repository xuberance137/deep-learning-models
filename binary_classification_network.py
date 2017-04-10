from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

PLOT_TRAINING=0

def vectorize_sequences(sequences, dimensions=10000):
	results = np.zeros((len(sequences), dimensions))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1
	return results

# get data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(nb_words=10000)

word_index = imdb.get_word_index()
reverser_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 0,1,2 reserved for padding, start of sequence and unknown
for n in range(10):
	decoded_review = ''.join([reverser_word_index.get(i-3, '*')+' ' for i in train_data[n]])
	print decoded_review

#vectorize/ ETL
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
#build network
# reset_selective model
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=10000))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=RMSprop(lr=0.001), loss=binary_crossentropy, metrics=[binary_accuracy])
#train network
history = model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512, validation_data=(x_val,y_val))
if PLOT_TRAINING:
	loss_values = history.history['loss']
	val_loss_values = history.history['val_loss']
	epochs = range(1, len(loss_values)+1)
	plt.plot(epochs, loss_values, 'bo')
	plt.plot(epochs, val_loss_values, 'b.')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.show()
#evaluate network
results = model.evaluate(x_test, y_test)
print 'Accuracy on test data set : ', results[1]
#predict with network
pred = model.predict(x_test)




