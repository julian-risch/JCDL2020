#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:22:23 2019

@author: Samuele Garda
"""


import tensorflow as tf
from models.abstract_model import AbstractModel
from models.components import embedding_layer

class SentenceCNN(AbstractModel):
  """
  Convolutional Neural Network for Sentence classification
  
  Implemented as https://arxiv.org/abs/1408.5882
  
  It applies a set of convolutional layers with different kernel sizes to learn features.
  """
  
  def __init__(self,params,train):
    """
    Initialize layers to build SentenceCNN model.
    
    Args:
      params (dict): hyperparameter defining layer sizes, dropout values, etc.
      train (bool): boolean indicating whether the model is in training mode. Used to
        determine if dropout layers should be added.
    """
    
    self.word_embd = embedding_layer.Embeddings(name = "txt_embd",W = params.pop("txt_embd"),
                                                hidden_size = params["hidden_size"],
                                                project = False)
    
    self.params = params
    self.train = train
    
    self.conv_filters = [tf.layers.Conv1D(filters = params["hidden_size"],
                                          kernel_size = kernel_size,
                                          padding='same',
                                          data_format='channels_last',
                                          activation = tf.nn.relu) for kernel_size in params["kernel_sizes"]]
    
    self.mlp = tf.layers.Dense(units = params["hidden_size"],
                               use_bias=True,name="bottle_neck",
                               )
    
    self.project = tf.layers.Dense(units = params["vocab_size"],
                                   use_bias=True,name="pre_softmax")
  
  
  def compute_loss(self,logits,labels):
    """
    Compute loss with sparse cross entropy.
    """
    
    with tf.name_scope("loss"):
      
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels)
      loss = tf.reduce_mean(losses)
    
      return loss
    
  def get_predictions(self,logits):
    """
    Compute top three classes from logits.
    """
    
    top_k_pred,top_k_ind = tf.nn.top_k(tf.nn.softmax(logits), k = 3)
    
    return top_k_ind
  
  def get_eval_metrics_op(self,logits,labels):
    """
    Compute accuracy.
    """
    
    predictions = tf.argmax(tf.nn.softmax(logits), 1)
    eval_metrics_ops = {"metrics/accuracy" : tf.metrics.accuracy(predictions,labels)}
    
    return eval_metrics_ops


  
  def __call__(self,inputs,targets = None):
    """
    Compute logits with SentenceCNN model.
    
    Args:
      inputs (Tensor): sequence of ints with shape [batch_size, input_length]
    
    Return:
      logits :  unscaled probabilities [batch_size, num_classes]
    """
    
    with tf.variable_scope("SentenceCNN"):
      
      inputs.set_shape([inputs.shape.as_list()[0],self.params["max_len"]])
      
      embd = self.word_embd(inputs)
      
      pre_outs = []
      
      conv_out = embd
      
      for i,conv_filter in enumerate(self.conv_filters):
        
        conv_out = conv_filter(conv_out)
        pool_out = tf.layers.MaxPooling1D(pool_size = 2,padding = 'same', strides = 1,
                                          name = "pool_{}".format(i))(conv_out)
        flat_out = tf.layers.Flatten()(pool_out)
                
        pre_outs.append(flat_out)
      
      pre_out = tf.concat([*pre_outs],axis = 1)
            
      if self.train:
        pre_out = tf.nn.dropout(pre_out, 1.0 - self.params["dropout"])
        
      mlp_pre_out = self.mlp(pre_out)
            
      if self.train:
        mlp_pre_out = tf.nn.dropout(mlp_pre_out, 1.0 - self.params["dropout"])
        
      logits = self.project(mlp_pre_out)
      
      return logits
      
    
    
    
    
  