#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:02:39 2019

@author: Samuele Garda
"""

import logging
import smart_open
import tensorflow as tf

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')

class TFRecordWriter(object):
  """
  Write tfrecords files.
  """
  
  def __init__(self,tp,lp):
    """
    Initialize TFRecordWriter.
    
    Args:
      tp (pp_utils.TextPreprocess) : text preprocessing class
      lp (pp_utils.LabelPreprocess) : label preprocessing class
    """
    self.tp = tp
    self.lp = lp
  
  def _int64_feature(self,value):
    """
    Map list of ints to tf compatible Features
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
  
  
  def write_training_example(self,writer,text,w2i,label,l2i,seq_l2i):
    """
    Write training example to tfrecord file. 
    The example is stored as a dictionary containing:
      - text : list of words indices
      - len_text : length of text
      - label : int label
      - seq_label : list of sequence label indices
      - len_seq_label : length of text sequence label
    
    Args:
      writer (tf.io.TFRecordWriter) : tfrecords writer
      text (list) : list of words indices
      w2i (dict) : lookup (word -> index)
      label (str) : label
      l2i (dict) : lookup (label -> index)
      seq_l2i (dict) : lookup (label -> index)
    """
    
    text = self.tp.text2ids(text,w2i)
    length_text = [len(text)]
    id_label = [self.lp.label2id(label,l2i)]
    seq_label = self.lp.seq_label2ids(label,seq_l2i)
    length_seq_label = [len(seq_label)]

    
    features = {"text" : self._int64_feature(text),
                "len_text" : self._int64_feature(length_text),
                "label" : self._int64_feature(id_label),
                "seq_label" : self._int64_feature(seq_label), 
                "len_seq_label" : self._int64_feature(length_seq_label)}
    
    example = tf.train.Example(features=tf.train.Features(feature=features))
    
    writer.write(example.SerializeToString())
    
    
  def write_file_tfrecord(self,in_file,out_file,w2i,l2i,seq_l2i):
    """
    Write single tfrecords file.
    
    Args:
      in_file (str) : path to compressed file (txt.gz) containing in each line an example (text and label)
      out_file (str) : path to tfrecord file
      w2i (dict) : lookup (word -> index)
      l2i (dict) : lookup (label -> index)
      seq_l2i (dict) : lookup (label -> index)
    """
    
    writer = tf.io.TFRecordWriter(out_file)
    
    n_examples = 0
    
    with smart_open.open(in_file) as infile:
      
      for idx,line in enumerate(infile):
        
        text,labels = line.strip("\n").split("\t")
        
        try:
          first_label = labels.split()[0]
                  
          self.write_training_example(writer = writer,
                                      text = text,
                                      w2i = w2i,
                                      label = first_label,
                                      l2i = l2i,
                                      seq_l2i = seq_l2i)
          
          n_examples += 1
          
          if (idx%10000)==0:
            
            logger.info("Written {} examples to {}".format(idx,out_file))
        except IndexError:
          logger.warning("Error parsing example {} - labels : {}".format(labels))
    
    return n_examples


class TFRecordParser(object):
  """
  Read data from tfrecord files.
  """
  
  def __init__(self):
    """
    Initialiaze TFRecordParser.
    """
    self.sos = [1] 
  
  def sparsefeature2dense(self,sparse_item,shape):
    """
    Map sparse tensor to dense one with its original shape.
    
    Args:
      sparse_item (tf.sparse.SparseTensor) : sparse tensor read from tfrecord file
      shape (list) : tensor shape
    """
    
    shape = tf.cast(shape,tf.int32)
    
    item = tf.reshape(tf.sparse.to_dense(sparse_item),shape)
    
    return item
  
  
  def parse_training_instance(self,proto,max_len,add_sos):
    """
    Parse single example from tfrecord file.
    
    Args:
      proto () : tfrecords element
      max_len (int) : crop sequence to this value
      add_sos (bool) : add StartOfSentence index
    
    Return:
      text (Tensor) : sequence of ints with shape  [text length]
      label (Tensor) : int label []
      seq_label (Tensor) : sequence of ints with shape  [label sequence length]
    """
    
    features = {"text" :  tf.io.VarLenFeature(tf.int64),
                "len_text" : tf.io.FixedLenFeature((1,), tf.int64),
                "label" : tf.io.FixedLenFeature((), tf.int64),
                "seq_label" :tf.io.VarLenFeature(tf.int64),
                "len_seq_label" : tf.io.FixedLenFeature((1,), tf.int64)}
    
    parsed_features = tf.io.parse_single_example(proto, features)
    
    text = self.sparsefeature2dense(sparse_item = parsed_features["text"],
                                    shape = parsed_features["len_text"])[:max_len]
    
    
    label = parsed_features["label"]
    
    seq_label = self.sparsefeature2dense(sparse_item = parsed_features["seq_label"],
                                         shape = parsed_features["len_seq_label"])
    
    
    if add_sos:
      
      seq_label = tf.concat([self.sos,seq_label],axis = 0)
    
    
    return text,label,seq_label
  
  
  def create_dataset(self,paths,batch_size,mode,flat_label = True,epochs = 1,
                     shuffle = 1000,max_len = 200,add_sos = False, npc = 4):
    """
    Create dataset from tfrecord files:
    
      - batch and pad/crop text
      - pick type of label (flat or hierarchical sequence)
      - repeat dataset for `n` epochs
      
    Args:
      paths (list) : list of tfrecords files path
      batch_size (int) : mini batch size
      mode (str) : `train`,`dev`,`test`
      flat_label (bool) : return flat label instead of label sequence
      epochs (int) : number of epochs
      shuffle (int) : shuffle this amount of examples in dataset 
      max_len (int) : crop text if longer than this
      add_sos (bool) : add SOS index to text int sequence
      npc (int) : Number of Parallel Calls, i.e. number of cores to use for parsing tfrecords
      
    Return:
      dataset (tf.data.Dataset): parsed dataset to be used.
    
    """
    
    dataset = tf.data.TFRecordDataset(paths)
    
    dataset = dataset.map(lambda x : self.parse_training_instance(x,max_len,add_sos), num_parallel_calls = npc)
    
    if mode == 'train':
      
      dataset = dataset.shuffle(buffer_size = shuffle)
    
    if mode == 'train' or mode == 'dev':
      
      dataset = dataset.padded_batch(batch_size, padded_shapes = ([None],[],[None]) )  #if flat_label else dataset.padded_batch(batch_size, padded_shapes = (([None],[None]),()) )
      
      dataset = dataset.map(lambda  t,l,sl : (t,l),  num_parallel_calls = npc) if flat_label else dataset.map(lambda  t,l,sl : (t,sl) , num_parallel_calls = npc )
      
      dataset = dataset.repeat(epochs)
      
    elif mode == 'test':
      
      dataset = dataset.map(lambda  t,l,sl : t)
      
      dataset = dataset.padded_batch(batch_size, padded_shapes = ([None]))
      
    dataset = dataset.prefetch(batch_size)
    
    return dataset
      
    
if __name__ == "__main__":
  
  
  tf.compat.v1.enable_eager_execution()
  
  tfrp = TFRecordParser()
  
  infiles = ["./data/tfrecords_uspto2m/2013.tfrecords","./data/tfrecords_uspto2m/dev.tfrecords","./data/tfrecords_uspto2m/test.tfrecords"]
  dataset = tfrp.create_dataset(paths = infiles,
                                batch_size = 2,
                                mode = "train",
                                max_len = 10,
                                add_sos = True
                                )
  
#  def get_text_flat_label(x):
#    
#    return (x[0],x[2])
  
  dataset = dataset.map(lambda  t,l,sl : t )
  
  print(dataset)
  
  for idx,t in enumerate(dataset):
    if idx < 3:
      print(t)
    else:
      break
  