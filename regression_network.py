from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy

import numpy as np

def build_model():
	model = Sequential()
	model.add(Dense(64, activation='relu', input_dim=13)) #train_data.shape[1]
	model.add(Dense(1))
	model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
	return model

(train_data, train_target), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

# K fold validation
k=4
num_val_samples = len(train_data) // k
all_mae_histories = []
for n in range(k):
	print "\nProcessing fold ", n
	val_data = train_data[n*num_val_samples:(n+1)*num_val_samples]
	val_targets = train_target[n*num_val_samples:(n+1)*num_val_samples]
	partial_train_data = np.concatenate([train_data[:n*num_val_samples],train_data[(n+1)*num_val_samples:]], axis=0)
	partial_train_targets = np.concatenate([train_target[:n*num_val_samples],train_target[(n+1)*num_val_samples:]], axis=0)
	model = build_model()
	history = model.fit(partial_train_data, partial_train_targets, epochs=50, batch_size=1, verbose=0)
	val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
	all_mae_histories.extend(history.history['mean_absolute_error']) 

print all_mae_histories



