import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import numpy as np


def prep_data():
	imdb_dir = './data/aclImdb'
	train_dir = os.path.join(imdb_dir, 'train')

	labels = []
	texts = []

	for label_type in ['neg', 'pos']:
		dir_name = os.path.join(train_dir, label_type)
		for fname in os.listdir(dir_name):
			if fname[-4:] == '.txt':
				f = open(os.path.join(dir_name, fname))
				texts.append(f.read())
				f.close()
				if label_type == 'neg':
					labels.append(0)
				else:
					labels.append(1)

	return texts, labels

def parse_embeddings():
	glove_file = './data/glove.6B/glove.6B.100d.txt'
	embedding_index = {} #embedding dictionary word->vec

	f = open(glove_file)
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embedding_index[word] = coefs
	f.close()
	print('Found %s word vectors.', len(embedding_index))

	return embedding_index

if __name__ == '__main__':

	texts, labels = prep_data()
	embedding_index = parse_embeddings()

	maxlen = 100 # number of words used in each review
	training_samples = 200
	validation_samples = 10000
	max_words = 10000 # number of words used in the dataset
	embedding_dim = 100 #glove embedding dims

	tokenizer = Tokenizer(num_words=max_words)
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)
	word_index = tokenizer.word_index
	print('Found %s unique tokens', len(word_index))
	data = pad_sequences(sequences, maxlen=maxlen) #making sure that each sequence is same length
	labels = np.asarray(labels)
	print('Shape of data tensor : ', data.shape)
	print('Shape of label tensor : ', labels.shape)
	indices = np.arange(data.shape[0])
	np.random.shuffle(indices)
	data = data[indices]
	labels = labels[indices]

	x_train = data[:training_samples]
	y_train = labels[:training_samples]
	x_val = data[training_samples:training_samples+validation_samples]
	y_val = labels[training_samples:training_samples+validation_samples]

	# preparing glove word embedding matrix
	embedding_matrix = np.zeros((max_words, embedding_dim))
	for word, n in word_index.items():
		if n < max_words:
			embedding_vector = embedding_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[n] = embedding_vector

	
	model = Sequential()
	model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(Dense(1, activation='sigmoid')) #because you want binary classification
	model.summary()

	model.layers[0].set_weights([embedding_matrix]) # setting Embedding layer to the glove data for that 
	model.layers[0].trainable = False # freeze the Embedding Layer using pretrained glove data

	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
	history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
	model.save_weights('pretrained_glove_model.h5')






























