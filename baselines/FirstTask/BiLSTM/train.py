# ==============================================================================
# Function:
# Author:
# date:
# ==============================================================================
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
from myutils import *
import reader as reader
from model import SNLIModel
import numpy as np
from sklearn.metrics import confusion_matrix

flags = tf.flags
logging = tf.logging
#model
flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("save_path","model_saved","where the model is stored")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
#data
flags.DEFINE_string("data_path", "",
                    "Where the training/test data is stored.")
flags.DEFINE_string("train_file", "../../data/fir_task/train.txt", "")
flags.DEFINE_string("dev_file", "../../data/fir_task/dev.txt", "")
flags.DEFINE_string("test_file", "../../data/fir_task/test.txt", "")

#hyper-parameters

flags.DEFINE_float("init_scale", 0.1, "")
flags.DEFINE_float("min_lr", 0.000005, "")
flags.DEFINE_float("learning_rate", 0.0001,"")
flags.DEFINE_float("lr_decay",0.8 , "")
flags.DEFINE_integer("max_epoch", 8, "")
flags.DEFINE_integer("max_max_epoch", 4, "")
flags.DEFINE_integer("max_grad_norm", 5, "")
flags.DEFINE_integer("num_layers", 1, "")
flags.DEFINE_integer("maxlen", 50, "")
flags.DEFINE_integer("num_classes", 2, "")
flags.DEFINE_integer("hidden_units", 300, "")
flags.DEFINE_integer("embedding_size", 300, "")
flags.DEFINE_integer("MAXITER", 70, "")
flags.DEFINE_float("keep_prob", 0.7, "")
flags.DEFINE_integer("batch_size", 256, "")

flags.DEFINE_integer("vocab_size", 53522, "")
flags.DEFINE_float("l2_strength", 0.0003, "")
flags.DEFINE_integer("early_stopping", 10, "")
flags.DEFINE_float("best_accuracy", 0, "")
flags.DEFINE_integer("best_val_epoch", 0, "")
flags.DEFINE_integer("change_epoch", 5, "")
flags.DEFINE_integer("update_learning", 5, "")
FLAGS = flags.FLAGS

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

def fill_placeholder(data, model,flags):
  batch_x,batch_label,batch_x_mask, batch_x_len= data.next_batch(flags.batch_size)
  feed_dict = {model.x:batch_x , 
                model.label:batch_label,
                model.x_mask:batch_x_mask,
                model.x_len :batch_x_len,
                }

  return feed_dict

def run_epoch(session, data,model,flags, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  losses = 0.0
  iters = 0
  acc_total=0.0
  fetches = {
      "acc":model.acc,
      "loss": model.loss,
      "global_step":model.global_step,
      "learning_rate": model.learning_rate,
      "pred":model.pred
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op
  
  start_time = time.time()
  epoch_size = data.get_epoch_size(flags.batch_size)
  for step in range(epoch_size):
    feed_dict = fill_placeholder(data,model,flags)
    
    vals = session.run(fetches, feed_dict)
    acc = vals["acc"]
    loss = vals["loss"]
    global_step=vals["global_step"]
    pred = vals["pred"]

    learning_rate = vals["learning_rate"]

    losses += loss
    iters= iters+1
    acc_total += acc
  acc_average=acc_total/iters
  loss_average = losses/iters

  return acc_average,loss_average,global_step,learning_rate


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):

  Train,Dev,Test,vocab = reader.file2seqid(FLAGS)
  print("train size:", Train.get_data_num)
  print("dev size:", Dev.get_data_num)
  print("test size:", Test.get_data_num)
  print("vocab size:", len(vocab))
  print("Train._labels",Train._labels)
  print("Train._x",Train._x)
  
  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)

    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=tf.AUTO_REUSE, initializer=initializer):
        m = SNLIModel(is_training=True, flags=FLAGS)
    
    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = SNLIModel(is_training=False,flags=FLAGS)

    with tf.name_scope("Test"):
       with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = SNLIModel(is_training=False, flags=FLAGS)
    
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      t0=time.time()
      best_accuracy = FLAGS.best_accuracy
      best_val_epoch = FLAGS.best_val_epoch
      last_change_epoch = 0

      best_test_acc=0
      best_test_epoch=0

      for i in range(FLAGS.MAXITER):
        start_time=time.time()
        train_acc,train_loss,train_global_step,learning_rate = run_epoch(session,data=Train, model=m,flags=FLAGS, eval_op=m.optim, verbose=True)
        print("Epoch: %d train_acc: %.3f train_loss %.3f train_global_step:%s" % (i + 1,train_acc,train_loss,train_global_step))
        
        dev_acc,dev_loss,_,_= run_epoch(session,data=Dev,model=mvalid,flags=FLAGS)
        print("Epoch: %d dev_acc: %.3f dev_loss %.3f" % (i + 1, dev_acc,dev_loss))

        test_acc,test_loss,_,_ = run_epoch(session, data=Test,model=mtest,flags=FLAGS)
        print("Epoch: %d test_acc: %.3f test_loss %.3f" % (i + 1, test_acc,test_loss))
        sys.stdout.flush()
        # if <= then update 
        if best_accuracy <= dev_acc:
          best_accuracy = dev_acc
          best_val_epoch = i
          if FLAGS.save_path:
            print("train_global_step:%s.  Saving %d model to %s." % (train_global_step,i+1,FLAGS.save_path))
            sv.saver.save(session,FLAGS.save_path+"/model", global_step=train_global_step)
            print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

        if (i - best_val_epoch > FLAGS.update_learning)and(i-last_change_epoch>FLAGS.change_epoch):
          if learning_rate>FLAGS.min_lr:
            lr_decay = FLAGS.lr_decay ** max(i + 1 - FLAGS.max_epoch, 0.0)
            new_learning_rate = FLAGS.learning_rate * lr_decay
            last_change_epoch= i
            print("learning_rate-->change!Dang!Dang!Dang!-->%.10f"%(new_learning_rate))
            m.assign_lr(session,new_learning_rate)

        if best_test_acc <test_acc:
            best_test_acc= test_acc 
            best_test_epoch= i

        
        end_time=time.time()
        print("################# all_training time: %s one_epoch time: %s ############### " % ((end_time-t0)//60, (end_time-start_time)//60))
        if i - best_val_epoch > FLAGS.early_stopping:
          print ("best_val_epoch:%d  best_val_accuracy:%s"%(best_val_epoch+1,best_accuracy))
          print ("best_test_epoch:%d  best_test_accuracy:%s"%(best_test_epoch+1,best_test_acc))
          logging.info("Normal Early stop")
          print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
          break        
        elif i == FLAGS.MAXITER-1:
       
          print ("best_val_epoch:%d  best_val_accuracy:%s"%(best_val_epoch+1,best_accuracy))
          print ("best_test_epoch:%d  best_test_accuracy:%s"%(best_test_epoch+1,best_test_acc))
          logging.info("Finishe Training")
   
      
if __name__ == "__main__":
  tf.app.run()
