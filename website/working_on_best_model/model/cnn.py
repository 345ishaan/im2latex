import tensorflow as tf
import numpy as np
from ops import *

def CNN_Net(img_tensor):
	#Input Size is [batch, in_height, in_width, in_channels]
	ip = (img_tensor - 128.0) / 128.0

	h0 = tf.nn.relu(conv2d(ip,64,3,3,1,1, name='conv1'))
	h0 = tf.nn.max_pool(h0,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1')

	h1 = tf.nn.relu(conv2d(h0,128,3,3,1,1, name='conv2'))
	h1 = tf.nn.max_pool(h1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2')

	h2 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(h1,256,3,3,1,1, name = 'conv3'),scale=False))

	h3 = tf.nn.relu(conv2d(h2,256,3,3,1,1, name = 'conv4'))
	h3 = tf.nn.max_pool(h3,ksize=[1,1,2,1],strides=[1,1,2,1],padding='SAME',name='pool4')

	h4 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(h3,512,3,3,1,1, name = 'conv5'),scale=False))
	h5 = tf.nn.max_pool(h4,ksize=[1,2,1,1],strides=[1,2,1,1],padding='SAME',name='pool5')

	h6 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(h5,512,3,3,1,1, name = 'conv6'),scale=False))

	return h6




# def bn(x,mean,var,beta):
# 	return f.nn.batch_norm_with_global_normalization(x, mean, var, beta, scale_after_normalization=False)

