#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:20:29 2019

@author: Samuele Garda
"""

import tensorlfow as tf

class GLU(tf.layers.Layer):

   def __init__(self,kernel_size,filters,index,trainable):
      super(GLU,self).__init__()
      self.kernel_size = kernel_size
      self.filters = filters
      self.conv = tf.layers.Conv1D(kernel_size = self.kernel_size, filters = self.filters,
                                   name = "glu_conv_{}".format(index),
                                   trainable = trainable)
      self.gate = tf.layers.Conv1D(kernel_size = self.kernel_size, filters = self.filters, activation = "sigmoid",
                                   name = "glu_gate_{}".format(index),
                                   trainable = trainable)
      self.padding =  tf.keras.layers.ZeroPadding1D(padding = (self.kernel_size -1,0))
      self.mult = tf.keras.layers.Multiply()

   def call(self,x):
      pad_x = self.padding(x)
      out =  self.mult([self.conv(pad_x),self.gate(pad_x)])
      return out