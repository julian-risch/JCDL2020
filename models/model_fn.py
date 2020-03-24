#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:22:30 2019

@author: Samuele Garda
"""


import tensorflow as tf
from models import (transformer_encoder,transformer,
                    gru,leam,cnn,ral)

def get_model_class(model_type):
  """
  Retrieve class for constructing flat models (not seq2seq/hiearchical).
  
  Args:
    model_type (str) : type of model (i.e. name)
  
  Return:
    model_class (class) : class for instantiating model object
  
  """
  
  if model_type == "transformer_encoder":
    
    model_class = transformer_encoder.TransformerEncoder
  
  elif model_type == "transformer": 
    
    model_class = transformer.Transformer
    
  elif model_type == "gru":
    
    model_class = gru.GatedRecurrentNetwork
    
  elif model_type == "leam":
    
    model_class = leam.LEAM
    
  elif model_type == "cnn":
    
    model_class = cnn.SentenceCNN
  
  elif model_type == "ral":
    
    model_class = ral.ReadAttendLabel
  
  else:
    
    raise NotImplementedError("Class for model `{}` is not available yet".format(model_type))
  
  return model_class


def model_fn(features, labels, mode, params):
  """
  Defines how to train, evaluate and predict with the flat models (not seq2seq/hierarchical). 
  Used only with tf.estimator.Estimator class.
  
  Args:
    features (Tensor) : sequence of ints with shape [batch_size, input_length]
    labels (Tensor or None) : sequence of ints with shape [batch_size] or None
    mode (tf.estimator.ModeKeys) : determines what estimator should do
    params (dict): hyperparameter defining layer sizes, dropout values, etc.
  """
  
  with tf.variable_scope("model"):
  
    inputs, targets = features, labels 
    
    model_class = get_model_class(params["model_type"])
    
    model = model_class(params = params,
                        train = mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = model(inputs,targets)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = model.get_predictions(logits = logits)
      return tf.estimator.EstimatorSpec(mode = mode, predictions=predictions)
    
    with tf.name_scope("loss"):
      loss = model.compute_loss(logits = logits, labels = targets)
      
    if mode == tf.estimator.ModeKeys.EVAL:
      eval_metrics_op = model.get_eval_metrics_op(logits = logits, labels = labels)
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss,eval_metric_ops = eval_metrics_op)
    
    else:
      train_op = model.get_train_op(params = params,loss = loss)
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op = train_op)

#def transformer_model_fn(features, labels, mode, params):
#  """
#  Defines how to train, evaluate and predict with the transformer. 
#  Used only with tf.estimator.Estimator class.
#  
#  Args:
#    features (Tensor) : sequence of ints with shape [batch_size, input_length]
#    labels (Tensor or None) : sequence of ints with shape [batch_size, output_length] or None
#    mode (tf.estimator.ModeKeys) : determines what estimator should do
#    params (dict): hyperparameter defining layer sizes, dropout values, etc.
#  """
#  
#  with tf.variable_scope("model"):
#    
#    inputs, targets = features, labels
#    
#    model = transformer.Transformer(params = params,
#                                    train = mode == tf.estimator.ModeKeys.TRAIN)
#    
#    logits = model(inputs, targets)
#    
#    if mode == tf.estimator.ModeKeys.PREDICT:
#      return tf.estimator.EstimatorSpec(mode = mode, predictions=logits)
#    
#    logits.set_shape(targets.shape.as_list() + logits.shape.as_list()[2:])
#      
#    xentropy, weights = metrics.padded_cross_entropy_loss(logits, targets, params["label_smoothing"], params["vocab_size"])
#      
#    loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
#      
#    # Save loss as named tensor that will be logged with the logging hook.
#    tf.identity(loss, "cross_entropy")
#    
#    if mode == tf.estimator.ModeKeys.EVAL:
#      
#      return tf.estimator.EstimatorSpec(mode=mode, loss=loss,eval_metric_ops=metrics.get_eval_metrics(logits, labels, params))
#    
#    else:
#            
#      train_op = model.get_train_op(loss, params)
#            
#      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


#def flat_model_fn(features,labels,mode,params):
#  """
#  Defines how to train, evaluate and predict with the flat models (not seq2seq/hierarchical). 
#  Used only with tf.estimator.Estimator class.
#  
#  Args:
#    features (Tensor) : sequence of ints with shape [batch_size, input_length]
#    labels (Tensor or None) : sequence of ints with shape [batch_size] or None
#    mode (tf.estimator.ModeKeys) : determines what estimator should do
#    params (dict): hyperparameter defining layer sizes, dropout values, etc.
#  """
#  
#  with tf.variable_scope("model"):
#  
#    inputs, targets = features, labels 
#    
#    model_class = get_flat_model_class(params["model_type"])
#    
#    model = model_class(params = params,
#                        train = mode == tf.estimator.ModeKeys.TRAIN)
#    
#    logits = model(inputs)
#    
#    if mode == tf.estimator.ModeKeys.PREDICT:
#      top_k_pred,top_k_ind = tf.nn.top_k(tf.nn.softmax(logits), k = 3)
#      return tf.estimator.EstimatorSpec(mode = mode, predictions=top_k_ind)
#    
#    with tf.name_scope("loss"):
#      loss = model.compute_loss(logits = logits, labels = targets)
#      
#    if mode == tf.estimator.ModeKeys.EVAL:
#      predictions = tf.argmax(tf.nn.softmax(logits), 1)
#      eval_metrics_ops = {"metrics/accuracy" : tf.metrics.accuracy(predictions,labels)}
#      return tf.estimator.EstimatorSpec(mode=mode, loss=loss,eval_metric_ops = eval_metrics_ops)
#    
#    else:
#      train_op = model.get_train_op(params = params,loss = loss)
#      return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op = train_op)
#      

#def get_model_fn(model_type):
#  """
#  Retrieve model function for tf.estimator based on model.
#  
#  Args:
#    model_type (str) : type of model (i.e. name)
#  
#  Return:
#    model_fn (function) : function containing model operation for tf.estimator
#  """
#  
#  if model_type == "transformer":
#    
#    model_fn = transformer_model_fn
#      
#  elif model_type in ["gru","transformer_encoder","leam","cnn"]:
#  
#    model_fn =  flat_model_fn
#  
#  else:
#    
#    raise NotImplementedError("Estimator function for model `{}` is not available yet".format(model_type))
#  
#  return model_fn

