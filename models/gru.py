#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:30:47 2018

@author: Samuele Garda
"""

import tensorflow as tf
from models.abstract_model import AbstractModel
from models.components import embedding_layer


class BidirectionalGRU(tf.layers.Layer):
  """
  Bidirectional Gated Recurrent Unit
  
  Runs two GRU cells (forward/backward) to compute features
  """
  
  def __init__(self,hidden_size):
    """
    Instantiate layer.
    
    Args:
      hidden_size (int) : dimension of single cell
    """
    
    super(BidirectionalGRU, self).__init__()
    
    self.hidden_size = hidden_size
    
    with tf.variable_scope("bidirectional_gru"):
      self.cell_fw = tf.nn.rnn_cell.GRUCell(num_units = hidden_size, name = "fw")
      self.cell_bw = tf.nn.rnn_cell.GRUCell(num_units = hidden_size, name = "bw")
  
  def call(self,inputs,seqs_length):
    """
    Run Bidirectional Recurrent Neural Network with GRU cells. Concatenates final hidden states to
    generate final representation.
    
    Args:
      inputs (Tensor): continuous tensor (embedded input) [batch_size,length,embedding_dim]
    
    Return:
      
      out (Tensor) : concatenation of final states [batch_size,embedding_dim]
    """
        
    cell_fw = self.cell_fw
    cell_bw = self.cell_bw
    
    batch_size = tf.shape(inputs)[0]
    
    initial_state_fw = cell_fw.zero_state(batch_size = batch_size, dtype = tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size = batch_size, dtype = tf.float32)
        
    # make tensor time first for faster computation
    # [batch,time,embd] -> [time,batch,embd]
    inputs = tf.transpose(inputs, [1,0,2])
    
    outputs, final_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                            cell_bw = cell_bw,
                                                            initial_state_fw=initial_state_fw,
                                                            initial_state_bw = initial_state_bw,
                                                            inputs= inputs,
                                                            sequence_length= seqs_length,
                                                            dtype=tf.float32,
                                                            time_major = True)
    
    out = tf.concat([*final_states],axis = -1)
    
    return out
    
    
class GatedRecurrentNetwork(AbstractModel):
  """
  Gated Recurrent Network.
  
  It uses bidirectional GRU to learn features, applies dropout and has classification layer.
  """
  
  def __init__(self,params,train):
    """
    Initialize layers to build GatedRecurrentNetwork model.
    
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
    
    self.bigru = BidirectionalGRU(hidden_size = params["hidden_size"])
  
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
    Compute logits with GatedRecurrentNetwork model.
    
    Args:
      inputs (Tensor): sequence of ints with shape [batch_size, input_length]
    
    Return:
      logits :  unscaled probabilities [batch_size, num_classes]

    """
        
    with tf.variable_scope("GatedRecurrentNetwork"):
    
      train = self.train
      
      seqs_length = self.get_seqs_length(inputs)
      
      embd = self.word_embd(inputs)
      
      embd_do = tf.keras.layers.SpatialDropout1D(rate = self.params["embd_do"])(embd,train)
      
      gru_out = self.bigru(embd_do,seqs_length)
      
      if self.train:
        gru_out = tf.nn.dropout(gru_out, 1.0 - self.params["dropout"])
      
      logits = self.project(gru_out)
      
      return logits
    
    
if __name__ == "__main__":
  
  import numpy as np
  
  tf.enable_eager_execution()
  
  X = tf.cast(tf.convert_to_tensor(np.random.rand(2,10,100)),tf.float32)
  
  seqs_len = tf.convert_to_tensor(np.asarray([10,10]))
  
  bigru = BidirectionalGRU(hidden_size  = 128)
  
  out = bigru(X,seqs_len)
  
  print("Out: ",out.shape)
  
  
  