#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:48:50 2019

@author: Samuele Garda
"""

import tensorflow as tf
from models.transformer import EncoderStack
from models.components import embedding_layer
from models.components import transformer_utils as model_utils


class TransformerEncoder(object):
  """
  Transformer encoder.
  
  Encoder stack from original Transformer to learn features 
  with classification layer.
  """
  
  def __init__(self,params,train):
    """
    Initialize layers to build TransformerEncoder model.
    
    Args:
      params (dict): hyperparameter defining layer sizes, dropout values, etc.
      train (bool): boolean indicating whether the model is in training mode. Used to
        determine if dropout layers should be added.

    """
    
    self.word_embd = embedding_layer.Embeddings(name = "txt_embd",
                                                W = params.pop("txt_embd"),
                                                hidden_size = params["hidden_size"])
    
    self.params = params
    self.train = train
    
    self.encoder_stack = EncoderStack(params, train)
    
    self.project = tf.layers.Dense(units = params["vocab_size"],use_bias=True,name="pre_softmax")
    
    
  def encode(self, inputs, attention_bias):
    """Generate continuous representation for inputs.

    Args:
      inputs: int tensor with shape [batch_size, input_length].
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

    Returns:
      float tensor with shape [batch_size, input_length, hidden_size]
    """
    with tf.name_scope("encode"):
      # Prepare inputs to the layer stack by adding positional encodings and
      # applying dropout.
      embedded_inputs = self.word_embd(inputs)
      inputs_padding = model_utils.get_padding(inputs)

      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(embedded_inputs)[1]
        pos_encoding = model_utils.get_position_encoding(
            length, self.params["hidden_size"])
        encoder_inputs = embedded_inputs + pos_encoding

      if self.train:
        encoder_inputs = tf.nn.dropout(
            encoder_inputs, 1 - self.params["layer_postprocess_dropout"])

      return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)
    
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
  
  def get_learning_rate(self,learning_rate, hidden_size, learning_rate_warmup_steps):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    
    with tf.name_scope("learning_rate"):
      
      warmup_steps = tf.to_float(learning_rate_warmup_steps)
      step = tf.to_float(tf.train.get_or_create_global_step())
  
      learning_rate *= (hidden_size ** -0.5)
      # Apply linear warmup
      learning_rate *= tf.minimum(1.0, step / warmup_steps)
      # Apply rsqrt decay
      learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))
  
      # Create a named tensor that will be logged using the logging hook.
      # The full name includes variable and names scope. In this case, the name
      # is model/get_train_op/learning_rate/learning_rate
      tf.identity(learning_rate, "learning_rate")
  
      return learning_rate
    
  def get_train_op(self,loss, params):
  
    with tf.variable_scope("get_train_op"):
      
      learning_rate = self.get_learning_rate(learning_rate=params["learning_rate"],
                                             hidden_size=params["hidden_size"],
                                             learning_rate_warmup_steps=params["learning_rate_warmup_steps"])
      
      # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
      # than the TF core Adam optimizer.
      optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate,
                                                   beta1=params["optimizer_adam_beta1"],
                                                   beta2=params["optimizer_adam_beta2"],
                                                   epsilon=params["optimizer_adam_epsilon"])

      # Calculate and apply gradients using LazyAdamOptimizer.
      global_step = tf.train.get_global_step()
      tvars = tf.trainable_variables()
      gradients = optimizer.compute_gradients(
          loss, tvars, colocate_gradients_with_ops=True)
      minimize_op = optimizer.apply_gradients(gradients, global_step=global_step, name="train")
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      train_op = tf.group(minimize_op, update_ops)
      
      train_metrics = {"learning_rate": learning_rate}
  
      gradient_norm = tf.global_norm(list(zip(*gradients))[0])
      train_metrics["global_norm/gradient_norm"] = gradient_norm
      
      for key, value in train_metrics.items():
        tf.summary.scalar(name=key, tensor=value)
        
      return train_op

    
  def __call__(self,inputs,targets = None):
    """
    Compute logits with TransformerEncoder model.
    
    Args:
      inputs (Tensor): sequence of ints with shape [batch_size, input_length]
    
    Return:
      logits :  unscaled probabilities [batch_size, num_classes]

    """
        
    with tf.variable_scope("TransformerEncoder"):
      
      attention_bias = model_utils.get_padding_bias(inputs)
      
      encoder_outputs = self.encode(inputs, attention_bias)
            
      sent_vector = tf.keras.layers.GlobalAveragePooling1D()(encoder_outputs)
                  
      logits = self.project(sent_vector)
    
      return logits
    
