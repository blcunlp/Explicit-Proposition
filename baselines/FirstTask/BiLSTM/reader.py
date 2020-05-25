# coding: utf-8
import tensorflow as tf
import os
import json
from myutils import *
from collections import Counter
import pandas as pd
from six.moves import xrange
import numpy as np
import pkuseg

_PAD="_PAD"
_UNK= "_UNK"
_GO= "_GO"
_EOS= "_EOS"
_START_VOCAB=[_PAD,_UNK,_GO,_EOS]

PAD_ID=0
UNK_ID=1
GO_ID =2
EOS_ID =3

def filter_length(seq,maxlen):
  if len(seq)>maxlen:
    new_seq=seq[:maxlen]
  else:
    new_seq=seq
  return new_seq

def load_data(train,vocab,seg):
    X,Z=[],[]
    #f_l=open("seq.txt","w+")
    max_len = 0
    for c,p,l in train:
        l=str(l)
        p=map_to_idx(tokenize(p,seg),vocab)+ [EOS_ID]
        if len(p) >= max_len:
          max_len = len(p)
        p=filter_length(p,50)
        X+=[p]
        Z+=[l]
    return X,Z,max_len

def get_vocab(data,seg):
    vocab=Counter()
    for ex in data:
        tokens=[]
        for tok in ex[1]:
          if(tok !=""):
            tokens.append(tok)
        #tokens=tokenize(ex[1],seg)
        vocab.update(tokens)
    #lst = ["unk", "delimiter"] + [ x for x, y in vocab.iteritems() if y > 0]
    vocab_sorted = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))
    lst = _START_VOCAB + [ x for x, y in vocab_sorted if y > 0]

    vocab_exist=os.path.isfile("../../data/fir_task/char_vocab.txt")

    if not vocab_exist:
      print ("write vocab.txt")
      f =open("../../data/fir_task/char_vocab.txt","w+")
      for x,y in enumerate(lst):
        x_y = str(y) +"\t"+ str(x)+"\n"
        f.write(x_y)
        #f.write("\n")
      f.close()
    
    vocab = dict([ (y,x) for x,y in enumerate(lst)])
    return vocab


class DataSet(object):
  def __init__(self,x,labels,x_len,X_mask):
    self._data_len=len(x)
    self._x =x
    self._labels =labels
    self._x_len = x_len
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = x.shape[0]
    self._x_mask=X_mask

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    end = self._index_in_epoch

    batch_x, batch_x_mask, batch_x_len = self._x[start:end], self._x_mask[start:end], self._x_len[start:end]
    batch_labels = self._labels[start:end]
    
    return batch_x, batch_labels,batch_x_mask,batch_x_len

  @property
  def get_x(self):
    return self._x
  
  @property
  def get_y(self):
    return self._y

  @property
  def labels(self):
    return self._labels

  @property
  def get_x_len(self):
    return self._x_len
  
  @property
  def get_y_len(self):
    return self._y_len

  @property
  def get_data_num(self):
    return self._data_len
  
  def get_epoch_size(self,batch_size):
    epoch_size = self._data_len //batch_size
    return epoch_size

def file2seqid(flags):
  
  maxlen = flags.maxlen

  seg = pkuseg.pkuseg()
  train = [l.strip().split('\t') for l in open(flags.train_file)]
  dev = [l.strip().split('\t') for l in open(flags.dev_file)]
  test = [l.strip().split('\t') for l in open(flags.test_file)]
  
  vocab = get_vocab(train,seg)

  X_train, Z_train,max_len_train = load_data(train, vocab,seg)
  X_test, Z_test,max_len_test = load_data(test, vocab,seg)
  X_dev, Z_dev,max_len_dev = load_data(dev, vocab,seg)
  print("max_len_train:{}".format(max_len_train))
  print("max_len_dev:{}".format(max_len_dev))
  print("max_len_test:{}".format(max_len_test))
  X_train_lengths=np.asarray([len(x) for x in X_train]).reshape(len(X_train))
  X_dev_lengths = np.asarray([len(x) for x in X_dev]).reshape(len(X_dev))
  X_test_lengths = np.asarray([len(x) for x in X_test]).reshape(len(X_test))
  X_train_mask = np.asarray([np.ones(x) for x in X_train_lengths]).reshape(len(X_train_lengths))
  X_dev_mask= np.asarray([np.ones(x) for x in X_dev_lengths] ).reshape(len(X_dev_lengths))
  X_test_mask=np.asarray([np.ones(x) for x in X_test_lengths] ).reshape(len(X_test_lengths))

  Z_train = to_categorical(Z_train, num_classes=flags.num_classes)
  Z_dev = to_categorical(Z_dev, num_classes=flags.num_classes)
  Z_test = to_categorical(Z_test, num_classes=flags.num_classes)
  
  X_train = pad_sequences(X_train, maxlen=maxlen, value=vocab[_PAD], padding='post') ## NO NEED TO GO TO NUMPY , CAN GIVE LIST OF PADDED LIST
  X_dev = pad_sequences(X_dev, maxlen=maxlen, value=vocab[_PAD], padding='post')
  X_test = pad_sequences(X_test, maxlen=maxlen, value=vocab[_PAD], padding='post')

  X_train_mask=pad_sequences(X_train_mask, maxlen=maxlen, value=vocab[_PAD], padding='post')
  X_dev_mask = pad_sequences(X_dev_mask , maxlen=maxlen, value=vocab[_PAD], padding='post')
  X_test_mask = pad_sequences(X_test_mask , maxlen=maxlen, value=vocab[_PAD], padding='post')
  
  Train = DataSet(X_train,Z_train,X_train_lengths,X_train_mask)
  Dev = DataSet(X_dev,Z_dev,X_dev_lengths,X_dev_mask)
  Test = DataSet(X_test,Z_test,X_test_lengths,X_test_mask) 
  
  return Train,Dev,Test,vocab
  
  
def snli_producer(data,config,name=None):
  batch_size=config.batch_size
  xmaxlen=config.xmaxlen
  ymaxlen=config.ymaxlen
  num_classes=config.num_classes
  with tf.name_scope(name, "SNLIProducer"):
    data_len=data._data_len
    epoch_size =data_len//batch_size
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

    data_x = tf.convert_to_tensor(data._x)
    data_y = tf.convert_to_tensor(data._y)
    data_labels =  tf.convert_to_tensor(data._labels)
    data_x_len =  tf.convert_to_tensor(data._x_len)
    data_y_len =  tf.convert_to_tensor(data._y_len)
    data_x_mask=tf.convert_to_tensor(data._x_mask)
    data_y_mask=tf.convert_to_tensor(data._y_mask)

    x = tf.strided_slice(data_x, [i*batch_size,0], [(i+1)*batch_size,xmaxlen])
    y = tf.strided_slice(data_y, [i*batch_size,0], [(i+1)*batch_size,ymaxlen])

    x_mask=tf.strided_slice(data_x_mask, [i*batch_size,0], [(i+1)*batch_size,xmaxlen])
    y_mask=tf.strided_slice(data_y_mask, [i*batch_size,0], [(i+1)*batch_size,ymaxlen])

    label = tf.strided_slice(data_labels, [i*batch_size,0], [(i+1)*batch_size,num_classes])
    x_len = tf.strided_slice(data_x_len, [i*batch_size], [(i+1)*batch_size])
    y_len = tf.strided_slice(data_y_len, [i*batch_size], [(i+1)*batch_size])
    
    x.set_shape([batch_size,xmaxlen])
    y.set_shape([batch_size,ymaxlen])
    x_mask.set_shape([batch_size,xmaxlen])
    y_mask.set_shape([batch_size,ymaxlen])
    label.set_shape([batch_size,num_classes])
    x_len.set_shape([batch_size])
    y_len.set_shape([batch_size])
 
    x_mask=tf.cast(x_mask,tf.float32)
    y_mask=tf.cast(y_mask,tf.float32)

    return x,y,label,x_len,y_len,data_len,x_mask,y_mask

 
if __name__=="__main__":
  print("reader is doing...")
