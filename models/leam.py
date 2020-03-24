#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:57:34 2019

@author: Samuele Garda
"""

import tensorflow as tf
from models.abstract_model import AbstractModel
from models.components import embedding_layer

class MultiLayerPerceptron(tf.layers.Layer):
  """
  Classification layer with bottleneck relu layer.
  
  """
  
  def __init__(self,mlp_size,out_size,dropout,train):
    """
    Instantiate layer.
    
    Args:
      mlp_size (int) : dimension for bottleneck layer
      out_size (int) : number of classes
      dropout (float) : dropout rate
      train (bool) : model is in training phase
    """
    
    super(MultiLayerPerceptron, self).__init__()
        
    self.W1 = tf.layers.Dense(units = mlp_size,
                              use_bias=True,name="W1",
                              activation = tf.nn.relu,
                              )
    self.project = tf.layers.Dense(units = out_size,
                                   use_bias=True,name="pre_softmax",
                                   )
    
    self.train = train
    self.dropout = dropout
    
  def call(self,inputs):
    """
    Compute logits.
    
    Args:
      inputs (Tensor): continuous tensor [batch_size,feature_dim]
      
    Return:
      logits (Tensor) : unscaled probabilities [batch_size, num_classes]
    """
    
    pre_out = self.W1(inputs)
    
    if self.train:
      pre_out = tf.nn.dropout(pre_out, 1.0 - self.dropout)
      
    logits = self.project(pre_out)
    
    return logits
  
class CompatibilityAttention(tf.layers.Layer):
  """
  It implements compatibility attention for LEAM model.
  
  First it computes G, the dot-product between embedded input sequence and label embeddings matrix.
  Afterwards it computs V, i.e. it  uses sliding window (CNN filter) to compute attention over G with text portion.
  Finally it uses softmax to get attention weights and it uses them to weight original inputs sequence.
  """
  
  def __init__(self,out_size,ngram_size):
    """
    Instantiate layer.
    
    Args:
      out_size (int) : number of classes
      ngram_size (int) : size of text portion with which to comput attention
        
    """
    super(CompatibilityAttention, self).__init__()
    self.Att_v = tf.layers.Conv1D(filters = out_size,
                                  kernel_size= ngram_size,
                                  padding='same',
                                  data_format='channels_last',
                                  activation = tf.nn.relu)
    
  def call(self,inputs,x_mask,W_labels_trans):
    """
    Compute attented document vector.
    
    Args:
      inputs (Tensor) : continuous tensor (embedded input) [batch_size,length,embedding_dim]
      x_mask (Tensor) : mask for input [batch_size,length,1]
      W_labels_trans (Variable) : transposed embeddings matrix of labels [embedding_dim,num_classes] 
    """
    
    x_emb_1 = tf.multiply(inputs, x_mask)
    
    x_emb_norm = tf.nn.l2_normalize(x_emb_1, axis=2)  # b * s * e
            
    W_class_norm = tf.nn.l2_normalize(W_labels_trans, axis = 0) # e * c
            
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm) # b * s * c
        
    Att_v = self.Att_v(G)  #b * s *  c
                                     
    Att_v = tf.reduce_max(Att_v, axis=-1, keepdims=True) # b * s * 1
    
    Att_v_max = self._partial_softmax(logits = Att_v, weights = x_mask, dim = 1, name = 'Att_v_max') # b * s * 1
    
    x_att = tf.multiply(inputs, Att_v_max)
    
    H_enc = tf.reduce_sum(x_att, axis=1) 
        
    return H_enc
  
  def _partial_softmax(self,logits, weights, dim, name):
    """
    Compute attention weights with masked softmax.
    """
    
    with tf.name_scope('partial_softmax'):
      exp_logits = tf.exp(logits)
      if len(exp_logits.get_shape()) == len(weights.get_shape()):
          exp_logits_weighted = tf.multiply(exp_logits, weights)
      else:
          exp_logits_weighted = tf.multiply(exp_logits, tf.expand_dims(weights, -1))
      exp_logits_sum = tf.reduce_sum(exp_logits_weighted, axis=dim, keepdims=True)
      partial_softmax_score = tf.div(exp_logits_weighted, exp_logits_sum, name=name)
      
      return partial_softmax_score
  
  


class LEAM(AbstractModel):
  """
  Label-Embedding Attentive Model
  
  Implemented as https://arxiv.org/pdf/1805.04174.pdf
  
  Leverage both word and labels embeddings with attention model to generate document representation.
  """
  
  def __init__(self,params,train):
    """
    Initialize layers to build LEAM model.
    
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
    
    self.w_label = tf.get_variable(name = "w_label", initializer = self.params.pop("label_embd"))
    
    self.comp_att = CompatibilityAttention(out_size = params["vocab_size"],
                                           ngram_size = params["ngram_size"])
    
    
    self.mlp = MultiLayerPerceptron(mlp_size = params["mlp_size"],
                                    out_size = params["vocab_size"],
                                    dropout = params["dropout"],
                                    train = train)
    
  
  def compute_loss(self,logits,labels):
    """
    Compute loss with sparse cross entropy and regularization loss, i.e. it uses two layer mlp
    to predict classes from label embedding matrix. 
    """
    
    with tf.name_scope("loss"):
      
      with tf.name_scope("standard_loss"):
        
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels)
        loss = tf.reduce_mean(losses)
        
        tf.summary.scalar("std_loss", loss)
      
      with tf.name_scope("regularization_loss"):
        pred_class_by_embd = self.mlp(self.w_label[:-1,:]) #last index is empty (added just for matrix creation)
        true_class_by_embd = tf.eye(self.params["vocab_size"],dtype=tf.float32)
        reg_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_class_by_embd, 
                                                                logits=pred_class_by_embd)
        reg_loss = tf.reduce_mean(reg_losses)
        
        tf.summary.scalar("reg_loss", reg_loss)
      
      return loss + reg_loss
  
    
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
    Compute logits with LEAM model.
    
    Args:
      inputs (Tensor): sequence of ints with shape [batch_size, input_length]
    
    Return:
      logits :  unscaled probabilities [batch_size, num_classes]
    """
    
    with tf.variable_scope("LEAM"):
      
      x_mask = tf.expand_dims(self.get_seqs_mask(inputs), axis=-1)
      x_embd = self.word_embd(inputs)
      
      attended_x = self.comp_att(inputs = x_embd,x_mask = x_mask,
                                 W_labels_trans = tf.transpose(self.w_label,[1,0]))
      
      if self.train:
        attended_x = tf.nn.dropout(attended_x, 1.0 - self.params["dropout"]) 
    
      logits = self.mlp(attended_x)
      
      return logits
      
      
      
    
    
    