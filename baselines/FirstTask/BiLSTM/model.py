################
#20170918
# implement of enhanced LSTM, small modify
# the input is deliver by feed_dict
################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import inspect
import logging
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.layers import batch_norm,l2_regularizer
from tensorflow.python.ops import variable_scope

from ops_cudnn_bilstm import *

from myutils import *
import reader as reader

class SNLIModel(object):
  """The SNLI model."""

  def __init__(self, is_training, flags):
    if is_training == True:
   	 batch_size = flags.batch_size
    else:  
   	 batch_size = 1
    self.flags = flags
    self.is_training = is_training
    self.global_step = tf.Variable(0, trainable=False)

    self.learning_rate = tf.Variable(self.flags.learning_rate, trainable=False)
    self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self.learning_rate, self._new_lr)

    
    self.x = tf.placeholder(tf.int32, [self.flags.batch_size, self.flags.maxlen])

    self.x_mask = tf.placeholder(tf.int32, [self.flags.batch_size, self.flags.maxlen])
    self.x_mask = tf.cast(self.x_mask,tf.float32)

    self.x_len = tf.placeholder(tf.int32, [self.flags.batch_size,])
    self.x_len = tf.cast(self.x_len,tf.float32)

    self.label = tf.placeholder(tf.int32, [self.flags.batch_size,self.flags.num_classes])

    with tf.device("/cpu:0"):
      embedding = np.load("../../data/fir_task/char_embedding.npy")

      input_xemb = tf.nn.embedding_lookup(embedding, self.x)
    
      if is_training and self.flags.keep_prob < 1:
        input_xemb = tf.nn.dropout(input_xemb, self.flags.keep_prob)

    with tf.variable_scope("encode_x") as scope:
      encode_cudnn_cell = cudnn_lstm(num_layers=1,hidden_size=self.flags.hidden_units)
      self.x_output,_,_ = call_cudnn_lstm(inputs=input_xemb,num_layers=1,hidden_size=self.flags.hidden_units,cudnn_cell=encode_cudnn_cell)    


      if is_training and self.flags.keep_prob < 1:
        self.x_output = tf.nn.dropout(self.x_output, self.flags.keep_prob)
      print("scope.global_variables:",scope.global_variables())

      xx = self.x_output

    with tf.variable_scope("pooling"):
      v_x_max = tf.reduce_max(xx,axis=1)  #(b,2h)    
      v_x_sum = tf.reduce_sum(xx, 1)  #(b,x_len.2*h) ->(b,2*h)
      v_x_ave = tf.div(v_x_sum, tf.expand_dims(self.x_len, -1)) #div true length

      self.v = tf.concat([v_x_ave,v_x_max],axis=-1) #(b,8*h)
      if is_training and self.flags.keep_prob < 1:
        self.v = tf.nn.dropout(self.v, self.flags.keep_prob)

    with tf.variable_scope("pred-layer"):
      fnn1 = self.fnn(input=self.x_output,
                      out_dim=self.flags.hidden_units,
                      activation=tf.nn.tanh,
                      use_bias=True,
                      w_name="fnn-pred-W")

      if is_training and self.flags.keep_prob < 1:
        fnn1 = tf.nn.dropout(fnn1, self.flags.keep_prob)

      W_pred = tf.get_variable("W_pred", shape=[self.flags.hidden_units, self.flags.num_classes],regularizer=l2_regularizer(self.flags.l2_strength))

      self.pred = tf.nn.softmax(tf.matmul(fnn1, W_pred), name="pred")

      correct = tf.equal(tf.argmax(self.pred,1),tf.argmax(self.label,1))
      self.acc = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")
    
      self.loss_term = -tf.reduce_sum(tf.cast(self.label,tf.float32) * tf.log(self.pred),name="loss_term")
      self.reg_term = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),name="reg_term")
      self.loss = tf.add(self.loss_term,self.reg_term,name="loss")

    if not is_training:
        return
    with tf.variable_scope("bp_layer"):
      tvars = tf.trainable_variables()
      for var in tvars:
        print (var.name,var.shape)
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                      self.flags.max_grad_norm)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.optim = optimizer.apply_gradients(
          zip(grads, tvars),
          global_step=self.global_step)


  def fnn(self,input,out_dim,in_dim=None,activation=None,use_bias=False,w_name="fnn-W"):
     with tf.variable_scope("fnn-layer"):
       if in_dim==None:
         input_shape = input.get_shape().as_list()
         in_dim = input_shape[-1]

       W = tf.get_variable(w_name,shape=[in_dim,out_dim])
       out = tf.matmul(input,W)
 
       if use_bias == True:
         b_name = w_name + '-b'
         b = tf.get_variable(b_name, shape=[out_dim])
         out = out + b
 
       if activation is not None:
         out = activation(out)
     return out

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    
