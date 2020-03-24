#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:06:28 2019

@author: Samuele Garda
"""

import tensorflow as tf
from models.abstract_model import AbstractModel
from models.components import embedding_layer

class ReadAttendLabel(AbstractModel):
  """
  Read, Attend and Lable model.
  
  Model architecture is the same as Show, Attend and Tell, i.e. 
  
  implemented as in : https://arxiv.org/abs/1502.03044
    
  Encode text sequence as embedding matrix, then uses LSTM with Bahdanau attention
  to generate sequence of labels.
  """
  
  def __init__(self,params,train):
    """
    Initialize layers to build ReadAttendLabel model.
    
    Args:
      params (dict): hyperparameter defining layer sizes, dropout values, etc.
      train (bool): boolean indicating whether the model is in training mode. Used to
        determine if dropout layers should be added.
    """
    
    self.word_embd = embedding_layer.Embeddings(name = "txt_embd",W = params.pop("txt_embd"),
                                                hidden_size = params["hidden_size"], project = False)
    self.label_embd = embedding_layer.Embeddings(name = "label_embd",W = params.pop("label_embd"),
                                                 hidden_size = params["hidden_size"], project = False)
        
    self.params = params
    
    # add for compatibility with seq2seq.sequence_loss
    # 2 : indices for SOS and PAD tokens
    self.params["vocab_size"] = self.params["vocab_size"] + 2 
    
    self.train = train
    
    self.init_h = tf.layers.Dense(units = params["hidden_size"], 
                                  use_bias = True, name = 'lstm_h', 
                                  activation = tf.nn.relu)
    self.init_c = tf.layers.Dense(units = params["hidden_size"], 
                                  use_bias = True, name = 'lstm_c', 
                                  activation = tf.nn.relu)
    
    self.project = tf.layers.Dense(units = params["vocab_size"], 
                                   use_bias = True, name = 'pre_softmax_w')
    
  
  def get_decoder_cell(self,lstm_init_state,memory,text_seqs_length):
    """
    Initialize LSTM decoder cell with recurrent dropout and BahdanauAttention attention mechanism.
    
    Args:
      lstm_init_state (tf.nn.rnn_cell.LSTMStateTuple) : initial state (h,c) of LSTM cell
      memory (Tensor) : input as encoded by encoder [batch_size,legth,embedding_sim]
      text_seqs_length (Tensor) : vector containing length of each element in memory [batch_size]
    
    Return:
      decoder_cell (tf.contrib.seq2seq.AttentionWrapper) : LSTM cell equipped with attention mechnism and recurrent dropout
      init_state (tf.contrib.seq2seq.AttentionWrapperState) : initial state of LSTM cell with attention
    """
    
    with tf.variable_scope("decoder_cell"):
      
      batch_size = self.params["batch_size"]
      beam_width = self.params["beam_width"]
      dropout = 1 - self.params["dropout"]
      hidden_size = self.params["hidden_size"]
      att_size = self.params["attention_size"]
            
      with tf.variable_scope("lstm_cell"):
        
        # get simple lstm cell
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
                
    
        embd_size = self.label_embd.W.shape[1]
                    
        # create cell with dropout wrapper
        lstm_cell =  tf.nn.rnn_cell.DropoutWrapper(cell = lstm_cell,
                                                   input_keep_prob= dropout if self.train else 1.0,
                                                   output_keep_prob= dropout if self.train else 1.0,
                                                   state_keep_prob= dropout if self.train else 1.0,
                                                   variational_recurrent= True,
                                                   # need this for input_size parameter 
                                                   # when using variational recurrent dropout
                                                   input_size = hidden_size + embd_size, 
                                                   ###############
                                                   seed = 42,
                                                   dtype = tf.float32)
      
      # tile the first dimension of memory (encoder output)
      # to make it work with beam search
      with tf.name_scope("tile_memory_for_beam"):        
        if not self.train:
          memory = tf.contrib.seq2seq.tile_batch(memory,beam_width)
          text_seqs_length = tf.contrib.seq2seq.tile_batch(text_seqs_length,beam_width)
        
      attention = tf.contrib.seq2seq.BahdanauAttention(num_units = att_size,
                                                       memory = memory,
                                                       memory_sequence_length = text_seqs_length)
    
    
      decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell = lstm_cell,
                                                         attention_mechanism = attention,
                                                         attention_layer_size = att_size,
                                                         alignment_history = True)
      
      # tile the first dimension of lstm initial state
      # to make it work with beam search
      with tf.name_scope("tile_init_state_for_beam"):
        if not self.train:
          lstm_init_state = tf.contrib.seq2seq.tile_batch(lstm_init_state, beam_width)
          init_state = decoder_cell.zero_state(batch_size * beam_width, dtype = tf.float32)
        else:
          init_state = decoder_cell.zero_state(batch_size = batch_size, dtype = tf.float32)
        
        # use lstm initial state as initial state for cell with attention mechanism
        init_state.clone(cell_state = lstm_init_state)
        
      return decoder_cell,init_state
    
    
  
  def avg_embd(self,inputs,lengths):
    """
    Compute average of word embeddings considering length of each segement
    
    Args:
      inputs (Tensor) : embedded input [batch_size,length,embedding_size]
      length (Tensor) : vector containing length of each element in memory [batch_size]
    
    Return:
      avg_embd (Tensor) : average of word embeddings [batch_size,embedding_size]
    """
    with tf.name_scope("avg_embd"):
      sums = tf.reduce_sum(inputs,axis = 1)
      lenghts = tf.cast(tf.expand_dims(lengths, -1),tf.float32)
      avg_embd = tf.math.divide(sums, lenghts)
      
      return avg_embd
  
  def mlp_lstm_init_state(self,embd,text_length):
    """
    Generate LSTM decoder cell initial state, i.e. hidden state `h` 
    and inner memory `c` with two relu layers from average of 
    word embddings.
    
    Args:
      embd (Tensor) : embedded input [batch_size,length,embedding_size]
      text_length (Tensor) : vector containing length of each element in memory [batch_size]
      
    Return:
      init_state (tf.nn.rnn_cell.LSTMStateTuple) : initial state for LSTM cell
    """
    
    with tf.variable_scope("init_lstm_cell_state"):
      avg_embd = self.avg_embd(embd,text_length)
      init_h = self.init_h(avg_embd)
      init_c = self.init_c(avg_embd)
      init_state = tf.nn.rnn_cell.LSTMStateTuple(h = init_h, c = init_c)
      return init_state
    
  
  def get_sampling_prob(self,global_step,k,min_value = 0,max_value = 1):
    """
    Compute sampling probability for Scheduled Sampling
    
    Implemented as  in https://arxiv.org/abs/1506.03099
    
    It uses curriculum learning strategy to progressively increase simpling 
    following inverse sigmoid function.
    """
    
    with tf.variable_scope("InverseSigmoidDecay",[global_step, k, min_value, max_value]):
      
      result = k / (k + tf.exp(global_step / k))
      result = result * (max_value - min_value) + min_value
      result = tf.cast(result, tf.float32)

      return result

    
  
  def decode(self,decoder_cell,targets,initial_state):
    """
    Generate sequence with decoder cell during training using Teacher Forcing.
    If sample parameter is set use Scheduled Sampling instead.
    
    Args:
      decoder_cell (tf.contrib.seq2seq.AttentionWrapper) : LSTM cell equipped with attention mechnism and recurrent dropout
      init_state (tf.contrib.seq2seq.AttentionWrapperState) : initial state of LSTM cell with attention
    
    Return:
      logits (Tensor) : unscaled probabilities for sequence [batch_size,seq_legth,vocab_size]
    """
    
    embd_labels = self.label_embd(targets[:,:-1])
    w_labels = self.label_embd.W
    label_seqs_length = self.get_seqs_length(targets[:,1:])
    
    if not self.params["sample"]:
            
      train_helper = tf.contrib.seq2seq.TrainingHelper(inputs = embd_labels,
                                                       sequence_length = label_seqs_length)
    else:
      
      sampling_prob = self.get_sampling_prob(global_step = tf.train.get_global_step(),
                                             k = self.params["sampling_rate"])
      
      train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(inputs = embd_labels,
                                                                         sequence_length = label_seqs_length,
                                                                         embedding = w_labels,
                                                                         sampling_probability = sampling_prob,
                                                                         seed = 42)
      
    train_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell, 
                                                    helper = train_helper,
                                                    initial_state = initial_state,
                                                    output_layer = self.project)
    
    final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder = train_decoder,
                                                                                           maximum_iterations = self.params["max_iteration"],
                                                                                           impute_finished = True)
    
    
    logits = final_outputs.rnn_output
    
    return logits
  
  def predict(self,decoder_cell,initial_state):
    """
    Generate sequence at inference time with BeamSearch algorithm
    
    Args:
      decoder_cell (tf.contrib.seq2seq.AttentionWrapper) : LSTM cell equipped with attention mechnism and recurrent dropout
      init_state (tf.contrib.seq2seq.AttentionWrapperState) : initial state of LSTM cell with attention
    
    Return:
      predictions (Tensor) : sequences generated with BeamSearch [batch_size, beam_width,seq_length]
      alignment_history (Tensor) : attention weights for top generated sequence [batch_size,beam_width]

    """
    
    w_labels = self.label_embd.W
    
    end_token = tf.constant(self.params["end_token"], dtype = tf.int32)
    start_tokens = tf.tile(tf.constant([self.params["start_token"]], dtype=tf.int32), [self.params["batch_size"]])
        
    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell = decoder_cell,
                                                             embedding = w_labels,
                                                             start_tokens = start_tokens,
                                                             end_token = end_token,
                                                             initial_state = initial_state,
                                                             beam_width = self.params["beam_width"],
                                                             output_layer= self.project,
                                                             length_penalty_weight = self.params["length_penalty"],
                                                             coverage_penalty_weight = self.params["coverage_penalty"])
    
    final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder = inference_decoder,
                                                                                           maximum_iterations =  self.params["max_iteration"],
                                                                                           impute_finished = False)
    alignment_history = final_state.cell_state.alignment_history[:,0,:]
    
    # [batch,length,beam_width] -> [batch,beam_width,length]
    predictions = tf.transpose(final_outputs.predicted_ids, [0,2,1])
    
    return predictions,alignment_history
  
  
  def compute_loss(self,logits,labels):
    """
    Compute loss with sequence to sequence loss ( to be used for back propagation through time)
    """
    
    with tf.name_scope("loss"):
      
      label_seqs_mask = self.get_seqs_mask(labels[:,1:])
      
      loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                              targets = labels[:,1:],
                                              weights = label_seqs_mask)
      return loss
    
  
  def get_predictions(self,logits):
    """
    Compute top three classes from sequence generated by BeamSearch.
    """
            
    predictions = logits[:,:3,2]
    
    return predictions
  
  
  def get_eval_metrics_op(self,logits,labels):
    """
    Compute accuracy.
    """
    
    raise NotImplementedError("Accuracy in Development mode is still not available")
    
#    accuracy = metrics._convert_to_eval_metric(metrics.padded_accuracy)(logits, labels)
#    
#    eval_metrics_ops = {"metrics/accuracy" : accuracy}
#    
#    return eval_metrics_ops
    
  def __call__(self,inputs,targets = None):
    """
    Compute logits or get predictions for decoding a given sequence with ReadAttendLabel model.
    
    Args:
      inputs (Tensor): sequence of ints with shape [batch_size, input_length]
      targets (Tensor or None) : if None decode with BeamSearch else sequence of ints with shape [batch_size, output_legth]
    
    Return:
      logits (Tensor) : unscaled probabilities for sequence [batch_size,seq_legth,vocab_size]
      
        or
      
      predictions (Tensor) : sequences generated with BeamSearch [batch_size, beam_width,seq_length]
    """
    
    with tf.variable_scope("ReadAttendLabel"):
      
      train = self.train
      
      text_seqs_length = self.get_seqs_length(inputs)
            
      embd = self.word_embd(inputs)
            
      embd_do = tf.keras.layers.SpatialDropout1D(rate = self.params["embd_do"])(embd,train)
      
      lstm_init_state = self.mlp_lstm_init_state(embd_do,text_seqs_length)
            
      decoder_cell,init_state = self.get_decoder_cell(lstm_init_state = lstm_init_state,
                                                      memory = embd_do,
                                                      text_seqs_length = text_seqs_length)
      
      if targets is not None:
        
        logits = self.decode(decoder_cell = decoder_cell,initial_state = init_state,targets = targets)
        return logits
      
      else:
        # TODO : add in configuration file option for returning attention weights for visualization
        predictions,alignment_history = self.predict(decoder_cell = decoder_cell,initial_state = init_state)
        return predictions
        
      

if __name__ == "__main__":
  
  
#  tf.enable_eager_execution()    
  
  import json
  import numpy as np
  from io_utils import get_embeddings
  from train import get_nclasses
  
  config_path = "./data/configs/local_test/ral.json"
  
  config = json.load(open(config_path))   
  
  get_embeddings(config,config["in_dir"],config["model_type"])
  
  config["vocab_size"]= get_nclasses(config["in_dir"],config["model_type"])
  
  
  model = ReadAttendLabel(params = config, train = True)
  
  
#  inputs = np.asarray([[4949,4052,  518, 4034,  136, 4949, 4052,  518,  449, 2451 , 137, 8337,  791, 4949,
#  4052, 2347, 1097, 1355, 1757,  294],
# [ 107,   36 , 418,   19, 1390,   96,   18, 1390,  633,   71,   46,  112,    4,  241,
#     3,   17,   18, 1390,   96,   27]])
#  
#  targets = np.asarray([[1, 424, 515, 298,   2],
#            [1, 424, 236 ,593 ,  2]])
#  
  
  
  inputs = tf.placeholder(tf.int64,[None,30])
  targets = tf.placeholder(tf.int64,[None,5])
  res = model(inputs = inputs,targets = targets) 
  
