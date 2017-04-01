# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/ops.py
#   + License: MIT

import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

tfparam={}

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))
                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d",bias_flag=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if bias_flag == True:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            conv = tf.nn.bias_add(conv, biases)

        return conv

def conv2d_transpose(input_, output_shape,
                     k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                     name="conv2d_transpose", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def log_sum_exp(xs):
  maxes = tf.reduce_max(xs, keep_dims=True)
  xs -= maxes
  return tf.squeeze(maxes, [-1]) + tf.log(tf.reduce_sum(tf.exp(xs), -1))

def Linear(
    name,
    inputs,
    input_dim,
    output_dim,
    activation='linear',
    bias=True,
    init=None,
    weightnorm=False,
    **kwargs
    ):
    print input_dim
    print output_dim
    with tf.name_scope(name) as scope:
        weight = tf.get_variable(name+'.W',[input_dim,output_dim],tf.float32,tf.random_normal_initializer(stddev=1.0/np.sqrt(input_dim)))
        batch_size = None
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
            # nort.m_values = np.linalg.norm(weight_values, axis=0)

            target_norms = tf.get_variable(name + '.g',norm_values)

            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
                weight = weight * (target_norms / norms)

        if inputs.get_shape().ndims == 2:
            result = tf.matmul(inputs, weight)
        else:
            reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
            result = tf.matmul(reshaped_inputs, weight)
            result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))
        if bias:
            result = tf.nn.bias_add(result,bias=tf.get_variable(name+'.b',[output_dim],tf.float32,tf.constant_initializer(value=0,dtype=tf.float32)))
        return result


class decCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, name, n_in, n_hid, L, D, ctx, forget_bias=1.0):
        self._n_in = n_in
        self._n_hid = n_hid
        self._name = name
        self._forget_bias = forget_bias
        self._ctx = ctx
        self._L = L
        self._D = D

    @property
    def state_size(self):
        return self._n_hid

    @property
    def output_size(self):
        return self._n_hid

    def __call__(self, _input, state, scope=None):

        h_tm1, c_tm1, output_tm1 = tf.split(1,3,state)

        gates = Linear(
                self._name+'.Gates',
                tf.concat(1, [_input, output_tm1]),
                self._n_in + self._n_hid,
                4 * self._n_hid,
                activation='sigmoid'
            )

        i_t,f_t,o_t,g_t = tf.split(1, 4, gates)

        ## removing forget_bias
        c_t = tf.nn.sigmoid(f_t)*c_tm1 + tf.nn.sigmoid(i_t)*tf.tanh(g_t)
        h_t = tf.nn.sigmoid(o_t)*tf.tanh(c_t)


        target_t = tf.expand_dims(Linear(self._name+'.target_t',h_t,self._n_hid,self._n_hid,bias=False),2)
        # target_t = tf.expand_dims(h_t,2) # (B, HID, 1)
        a_t = tf.nn.softmax(tf.batch_matmul(self._ctx,target_t)[:,:,0],name='a_t') # (B, H*W, D) * (B, D, 1)
        
        
        a_t = tf.expand_dims(a_t,1) # (B, 1, H*W)
        z_t = tf.batch_matmul(a_t,self._ctx)[:,0]
        # a_t = tf.expand_dims(a_t,2)
        # z_t = tf.reduce_sum(a_t*self._ctx,1)

        output_t = tf.tanh(Linear(
            self._name+'.output_t',
            tf.concat(1,[h_t,z_t]),
            self._D+self._n_hid,
            self._n_hid,
            bias=False,
            activation='tanh'
            ))

        new_state = tf.concat(1,[h_t,c_t,output_t])

        return output_t,new_state

class LSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, name, n_in, n_hid, forget_bias=1.0):
        self._n_in = n_in
        self._n_hid = n_hid
        self._name = name
        self._forget_bias = forget_bias

    @property
    def state_size(self):
        return self._n_hid

    @property
    def output_size(self):
        return self._n_hid

    def __call__(self, inputs, state, scope=None):
        c_tm1, h_tm1 = tf.split(1,2,state)
        gates = Linear(
                self._name+'.Gates',
                tf.concat(1, [inputs, h_tm1]),
                self._n_in + self._n_hid,
                4 * self._n_hid,
                activation='sigmoid'
                )

        i_t,f_t,o_t,g_t = tf.split(1, 4, gates)

        c_t = tf.nn.sigmoid(f_t+self._forget_bias)*c_tm1 + tf.nn.sigmoid(i_t)*tf.tanh(g_t)
        h_t = tf.nn.sigmoid(o_t)*tf.tanh(c_t)

        new_state = tf.concat(1, [c_t,h_t])
        return h_t,new_state

def BiLSTM(name,ip,in_dim,hid_dim,prev_fw_state,prev_bw_state,batch_size):
    
    if prev_fw_state is None:
        prev_fw_state = tf.get_variable(name+'hidden_fw',np.zeros(2*hid_dim,dtype=np.float32))
        prev_fw_state = tf.reshape(tf.tile(prev_fw_state, tf.pack([batch_size])), tf.pack([batch_size, 2*hid_dim]))
    if prev_bw_state is None:
        prev_bw_state = tf.get_variable(name+'hidden_bw',np.zeros(2*hid_dim,dtype=np.float32))
        prev_bw_state = tf.reshape(tf.tile(prev_bw_state, tf.pack([batch_size])), tf.pack([batch_size, 2*hid_dim]))
    cell1 = LSTMCell(name+'_fw', in_dim, hid_dim)
    cell2 = LSTMCell(name+'_bw', in_dim, hid_dim)

    seq_len = tf.tile(tf.expand_dims(tf.shape(ip)[1],0),[batch_size])
    outputs = tf.nn.bidirectional_dynamic_rnn(cell1, cell2, ip, sequence_length=seq_len, initial_state_fw=prev_fw_state, initial_state_bw=prev_bw_state, swap_memory=True)
    return tf.concat(2,[outputs[0][0],outputs[0][1]])




def Embedding(name,n_symbols,output_dim,indices):
    with tf.name_scope(name) as scope:
        emb_weights = tf.get_variable(name,[n_symbols,output_dim],tf.float32,tf.random_normal_initializer(stddev=1.0/np.sqrt(n_symbols)))
        return tf.nn.embedding_lookup(emb_weights,indices)


def enc_init(name,H,enc_hid_dim):
    with tf.name_scope(name) as scope:
        return tf.get_variable(name,[1,H,2*enc_hid_dim],tf.float32,tf.constant_initializer(value=0,dtype=tf.float32)) # init for both memory and hidden state

def dec_init(name,dec_dim):
    with tf.name_scope(name) as scope:
        return tf.get_variable(name,[1,3*dec_dim],tf.float32,tf.constant_initializer(value=0,dtype=tf.float32))


def Attention_Enc(name,ip,hid_dim,W,H,D,batch_size):
    perm_conv_op = tf.transpose(ip,[0,1,2,3])
    hidden_state_init_fw = tf.tile(enc_init(name+'hidden_fw',H,hid_dim),[batch_size,1,1]) # The reason for 2*hid_dim is because one is for ct which is memory , other is for hidden
    hidden_state_init_bw = tf.tile(enc_init(name+'hidden_bw',H,hid_dim),[batch_size,1,1])
    def fn(prev_op,i):
        return BiLSTM(name+'.BiLSTMEncoder',perm_conv_op[:,i],D,hid_dim,hidden_state_init_fw[:,i],hidden_state_init_bw[:,i],batch_size)

    temp_enc = tf.scan(fn,tf.range(tf.shape(perm_conv_op)[1]), initializer=tf.placeholder(shape=(None,None,2*hid_dim),dtype=tf.float32))
    return tf.reshape(tf.transpose(temp_enc,[1,0,2,3]),[tf.shape(ip)[0],-1,hid_dim*2]) # output of encoder is Batch_Size, H*W, 512 =D

    

def Attention_Dec(name,ip,embedding,hid_dim,total_ct,D,batch_size,embedding_dim,vocab_size):

    hidden_state_init_dec = tf.tile(dec_init(name+'hidden_dec',hid_dim),[batch_size,1])
    cell = decCell(name+'.AttentionCell',embedding_dim,hid_dim,total_ct,D,ip)
    seq_len = tf.tile(tf.expand_dims(tf.shape(embedding)[1],0),[batch_size])
    out,state = tf.nn.dynamic_rnn(cell, embedding, initial_state=hidden_state_init_dec, sequence_length=seq_len, swap_memory=True)
    logits = Linear('logits',out,hid_dim,vocab_size)
    return out,logits

    