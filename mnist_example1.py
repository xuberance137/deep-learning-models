from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.utils.np_utils import to_categorical

# get data
((train_images, train_labels), (test_images, test_labels)) = mnist.load_data()
# ETL
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000, 28*28))
train_images = train_images.astype('float32')/255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# build network
network = Sequential()
network.add(Dense(512, activation='relu', input_shape=(28*28, )))
network.add(Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# train network
network.fit(train_images, train_labels, epochs=8, batch_size=128)
# test network
test_loss, test_acc = network.evaluate(test_images, test_labels)
print '\n', test_loss, '\t', test_acc, '\n'
