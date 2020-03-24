#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:29:12 2018

@author: Samuele Garda
"""

import tensorflow as tf
from abc import ABCMeta,abstractmethod

class AbstractModel(object, metaclass = ABCMeta):
  """
  Abstract class for NN models. 
  It contains methods used by all flat models (not seq2seq/hierarchical)
  """
  
  @abstractmethod
  def compute_loss():
    pass
  
  @abstractmethod
  def get_eval_metrics_op():
    pass
  
  @abstractmethod
  def get_predictions():
    pass
  
  
  def get_seqs_mask(self,inputs):
    """
    Create mask for sequence of ints, where 0 stands for padding
    
    Args:
      inputs (Tensor) : sequence of ints [batch_size,length]
      
    Return:
      seq_mask (Tensor) : mask [batch_size,length]
    """
    
    seqs_mask = tf.not_equal(inputs,0)
    seqs_mask = tf.cast(seqs_mask,tf.float32)
    
    return seqs_mask
  
  def get_seqs_length(self,inputs):
    """
    Create vector containing length of each element in batch
    
    Args:
      inputs (Tensor) : sequence of ints [batch_size,length]
      
    Return:
      seq_length (Tensor) : mask [batch_size]
    """
    
    seqs_mask = self.get_seqs_mask(inputs)
    seqs_length = tf.reduce_sum(tf.cast(seqs_mask,tf.int32),axis = 1)
    
    return seqs_length    


  def clip_and_step(self,optimizer, loss, clipping):
    """
    Compute gradients and clip them by global norm (optional).
    
    Args:
      
      optimizer (tf.train.Optimizer) : instance of optimizer, e.g. AdamOptimizer
      loss (Tensor) : loss computed with model (scalar) []
      clipping (int) : value of global norm to clip gradients with (0 disables it)
    """
  
    grads_and_vars = optimizer.compute_gradients(loss)
    grads, varis = zip(*grads_and_vars)
          
    if clipping:
        grads, global_norm = tf.clip_by_global_norm(grads, clipping,
                                                    name="gradient_clipping")
    else:
        global_norm = tf.global_norm(grads, name="gradient_norm")
    grads_and_vars = list(zip(grads, varis))  # list call is apparently vital!!
    train_op = optimizer.apply_gradients(grads_and_vars,
                                         global_step=tf.train.get_global_step(),
                                         name="train_step")
    return train_op, grads_and_vars, global_norm
  
  
  def record_scalars(self,metric_dict):
    for key, value in metric_dict.items():
      tf.summary.scalar(name=key, tensor=value)

  
  def get_train_op(self,loss,params):
    """
    Create training op needed for tf.estimator
    
    Args:
      loss (Tensor) : loss computed with model (scalar) []
      params (dict) : dictionary containing optimizer parameters
    
    Return:
      train_op (tf.Operation) : apply gradients update with tf.estimator
    """
    
    optimizer = tf.train.AdamOptimizer(learning_rate = params["learning_rate"],
                                      beta1=params["optimizer_adam_beta1"],
                                      beta2=params["optimizer_adam_beta2"],
                                      epsilon=params["optimizer_adam_epsilon"])
    
    train_op, grads_and_vars, global_grad_norm = self.clip_and_step(optimizer = optimizer, 
                                                                    loss = loss,
                                                                    clipping = params["clipping"])
    
    train_metrics = {}
    train_metrics["global_norm/gradient_norm"] = global_grad_norm
    
    self.record_scalars(train_metrics)
    
    return train_op
            
