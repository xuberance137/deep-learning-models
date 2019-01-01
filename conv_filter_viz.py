from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import keras.backend as K
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

'''
This can be done with gradient ascent in input space: applying gradient descent to the value of the input image of a convnet so as to maximize the response of a specific filter, starting from a blank input image. The resulting input image would be one that the chosen filter is maximally responsive to.
The process is simple: we will build a loss function that maximizes the value of a given filter in a given convolution layer, then we will use stochastic gradient descent to adjust the values of the input image so as to maximize this activation value.
'''

def generate_filter_viz(layer_name, filter_index, size=100):
	layer_output = model.get_layer(layer_name).output
	loss = K.mean(layer_output[:, :, :, filter_index])
	# The call to `gradients` returns a list of tensors (of size 1 in this case)
	# hence we only keep the first element -- which is a tensor.
	grads = K.gradients(loss, model.input)[0]
	 # We add 1e-5 before dividing so as to avoid accidentally dividing by 0.
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	iterate = K.function([model.input], [loss, grads])
	loss_value, grads_value = iterate([np.zeros((1, 100, 100, 3))])

	# We start from a gray image with some noise
	input_img_data = np.random.random((1, 100, 100, 3)) * 20 + 128.
	# Run gradient ascent for 40 steps
	step = 1.  # this is the magnitude of each gradient update
	for i in range(100):
		# Compute the loss value and gradient value
		loss_value, grads_value = iterate([input_img_data])
		# Here we adjust the input image in the direction that maximizes the loss
		input_img_data += grads_value * step

	img = input_img_data[0]
	return deprocess_image(img)

### MAIN FUNCTION ###
if __name__ == '__main__':
	model = VGG16(weights='imagenet', include_top=True)

	img_path = '/Users/gopal/Downloads/base5.jpg'
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	preds = model.predict(x)
	print('Predicted:', decode_predictions(preds, top=5)[0])

	for i, l in enumerate(model.layers):
		print(i, l.name, l.output_shape)

	layer_name = 'block3_conv1'
	filter_index = 200

	plt.imshow(generate_filter_viz(layer_name, filter_index))
	plt.show()


# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# import numpy as np

# model = ResNet50(weights='imagenet')

# img_path = '/Users/gopal/Downloads/base5.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# preds = model.predict(x)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=5)[0])