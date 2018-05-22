import os
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

def load_model(path):
	vgg = scipy.io.loadmat(path)
	vgg_layers = vgg['layers']
    
	def _weights(layer, expected_layer_name):
	    wb = vgg_layers[0][layer][0][0][2]
	    W = wb[0][0]
	    b = wb[0][1]
	    layer_name = vgg_layers[0][layer][0][0][0][0]
	    assert layer_name == expected_layer_name
	    return W, b

	    return W, b

	def _relu(conv2d_layer):
	    return tf.nn.relu(conv2d_layer)

	def _conv2d(prev_layer, layer, layer_name):
	    W, b = _weights(layer, layer_name)
	    W = tf.constant(W)
	    b = tf.constant(np.reshape(b, (b.size)))
	    return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

	def _conv2d_relu(prev_layer, layer, layer_name):
	    return _relu(_conv2d(prev_layer, layer, layer_name))

	def _avgpool(prev_layer):

	    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	# Constructs the graph model.
	graph = {}
	graph['input']   = tf.Variable(np.zeros((1, 400, 300, 3)), dtype = 'float32')
	graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
	graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
	graph['avgpool1'] = _avgpool(graph['conv1_2'])
	graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
	graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
	graph['avgpool2'] = _avgpool(graph['conv2_2'])
	graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
	graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
	graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
	graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
	graph['avgpool3'] = _avgpool(graph['conv3_4'])
	graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
	graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
	graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
	graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
	graph['avgpool4'] = _avgpool(graph['conv4_4'])
	graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
	graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
	graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
	graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
	graph['avgpool5'] = _avgpool(graph['conv5_4'])
	return graph


def white_noise(image, noise_ratio = 0.6):
	ni = np.random.uniform(-20, 20, (1, 400, 300, 3)).astype("float32")
	ni = noise_ratio*ni+(1-noise_ratio)*image
	return ni
def preprocess(image):
	#as vgg model requires
	image = np.reshape(image, (1,)+image.shape)
	return image

def save(path, image):
	image = np.clip(image[0],0,255).astype('uint8')
	scipy.misc.imsave(path, image)
	return 0

