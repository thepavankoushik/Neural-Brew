import os
import numpy as np
import tensorflow as tf
import scipy.io
import scipy.misc
import sys
import matplotlib.pyplot as plt
from utils import *
import cv2

#J(Generated) = alpha*J(content, Generated)+beta*J(style, Generated)
#JcontentG

def content_cost(a_c, a_g):
	m, n_h, n_w, n_c = a_g.get_shape().as_list()
	a_c = tf.reshape(a_c, [n_h*n_w, n_c])
	a_g = tf.reshape(a_g, [n_h*n_w, n_c])
	j = tf.reduce_sum(tf.square(tf.subtract(a_c, a_g)))
	j = j/(4*n_h*n_w*n_c)
	return j

#gram matrix for finding correlation between pixels in a particular layer
def gram_matrix(a):
	ga = tf.matmul(a, tf.transpose(a))
	return ga

#JstyleG
def layer_style_cost(a_s, a_g):
	m, n_h, n_w, n_c = a_g.get_shape().as_list()
	a_s = tf.reshape(a_s, [n_h*n_w, n_c])
	a_g = tf.reshape(a_g, [n_h*n_w, n_c])
	gs = gram_matrix(a_s)
	gg = gram_matrix(a_g)
	j = tf.reduce_sum(tf.square(tf.subtract(gs, gg)))
	j = j/(2*n_h*n_w*n_c)**2
	return j

#Jstyle computation from multiple layers
#layers and lamda values
style_layers = [('conv1_1',0.2),('conv2_1',0.2),('conv3_1',0.2),('conv4_1',0.2),('conv5_1',0.2)]
def style_cost(model, style_layers):
	j = 0
	for layer, lamda in style_layers:
		out = model[layer]
		a_s = sess.run(out)
		a_g = out
		j += lamda*layer_style_cost(a_s, a_g)
	return j


def cost(jc, js, alpha= 10, beta=40):
	j = (alpha*jc)+(beta*js)
	return j

tf.reset_default_graph()
sess = tf.InteractiveSession()

#content = scipy.misc.imread("lindan.jpg")
#content = preprocess(content)
#style = scipy.misc.imread("picasso.jpg")
#style = preprocess(style)

content = cv2.imread("lindan.jpg")
content = cv2.resize(content, dsize = (300, 400), interpolation = cv2.INTER_CUBIC)
content = preprocess(content)

style = cv2.imread("picasso.jpg")
style = cv2.resize(style, dsize = (300, 400), interpolation = cv2.INTER_CUBIC)
style = preprocess(style)

generated = white_noise(content)
model = load_model("vgg.mat")


#preparing jcontent tensor
sess.run(model["input"].assign(content))
out = model["conv4_2"]
a_c = sess.run(out)
a_g = out
jc = content_cost(a_c, a_g)


#preparing jstyle tensor
sess.run(model["input"].assign(style))
js = style_cost(model, style_layers)
j = cost(jc, js)

#using adam optimizer
optimizer = tf.train.AdamOptimizer(2.0).minimize(j)

def model_nn(sess, ip, iterations = 200):
	sess.run(tf.global_variables_initializer())
	sess.run(model["input"].assign(ip))
	for i in range(iterations):
		_= sess.run(optimizer)
		g = sess.run(model["input"])
		if(i%50==0):
			J, JC, JS = sess.run([j, jc, js])
			print(i, J, JC, jS)
			save("output/"+str(i)+".png", g)
	save("output/"+str(i)+".png",g)
	return g


model_nn(sess,generated)

