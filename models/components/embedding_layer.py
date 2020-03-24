# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Embeddings(tf.layers.Layer):
  """
  Map input sequence of ints to continuos space with embeddings vectors.
  It is possible to specify the desired output dimension. A linear layer is used 
  if the pre-trained embeddings do not match the given dimension.
  """
  
  def __init__(self,name,W,hidden_size,project = True):
    """    
    Args:
      hidden_size (int) : size of output embeddings
      W (np.ndarray) : pre-trained embeddings
      name (str) : for variable storing embeddings
      project (bool) : use linear layer to project embeddings to given dimension
    """
    super(Embeddings,self).__init__()
    self.hidden_size = hidden_size
    self.embeddings_name = name
    self.W = W
    self.project = project
  
  def build(self,_):
    """
    Build layer (add variables)
    """
    
    with tf.variable_scope("embd", reuse=tf.AUTO_REUSE):
      
      if self.project:
        in_channels = self.W.shape[1]
        
        if in_channels != self.hidden_size:
          self.p = tf.get_variable(name = "project",
                                   shape = [1,in_channels,self.hidden_size],
                                   initializer=tf.random_normal_initializer(0., self.hidden_size ** -0.5)
                                   )
        else:
          self.p = None
        self.W = tf.get_variable(name = self.embeddings_name, initializer = self.W)
      
    self.built = True
      
    
  def call(self,inputs):
    """
    Perform mapping from sequence of ints to continuos vectors.
    
    Args:
      inputs (Tensor) sequence of ints [batch_size,length]
    
    Return:
      embeddings : embedded input [batch_size,length,embedding_size]
    """
    
    mask = tf.to_float(tf.not_equal(inputs, 0))
    embeddings = tf.gather(self.W, inputs)
    if self.project:
      embeddings = tf.nn.conv1d(value = embeddings, filters = self.p, 
                                padding = "SAME",stride = 1)
    embeddings *= tf.expand_dims(mask, -1)
    embeddings *= self.hidden_size ** 0.5
        
    return embeddings
  
#  def linear(self, x):
#    """Computes logits by running x through a linear layer.
#    Args:
#      x: A float32 tensor with shape [batch_size, length, hidden_size]
#    Returns:
#      float32 tensor with shape [batch_size, length, vocab_size].
#    """
#    with tf.name_scope("presoftmax_linear"):
#      batch_size = tf.shape(x)[0]
#      length = tf.shape(x)[1]
#
#      x = tf.reshape(x, [-1, self.hidden_size])
#      logits = tf.matmul(x, self.W, transpose_b=True)
#
#    return tf.reshape(logits, [batch_size, length, self.vocab_size])