#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:04:27 2019

@author: Samuele Garda
"""

from gensim.parsing import preprocessing as pp

FILTERS = [lambda x : x.lower(), pp.strip_tags, pp.strip_punctuation, 
           pp.strip_multiple_whitespaces, pp.remove_stopwords,pp.strip_numeric, pp.strip_short]

class TextPreprocess:
  """
  Class containing helper function to preprocess text
  """
  
  
  @staticmethod
  def preprocess_string(string):
    """
    Use genim `preprocess_string` with custom filters
    
    Args:
      string (str) : input string
    Return:
      out (list) : preprocessed string
    """
    
    out = pp.preprocess_string(string,filters = FILTERS)
    return out
  
  @staticmethod
  def text2ids(txt_string,vocab):
    """
    Map string of words to list of indices.
    
    Args:
      txt_string (str) : preprocessed string (calling split returns tokenized string)
      vocab (dict) : lookup (word -> index)
    """
    
    return [vocab.get(w) for w in txt_string.split() if w in vocab]


def get_label_sequence(label):
  """
  From hierarchical label as sequence get all subpaths
  
  Args:
    label (str) : hierarhical label, in format "A.B.C"
  Return
    paths (list) : hierarchical subpaths
    
  >>>get_label_sequence("A.B.C")
  ["A","A.B","A.B.C"]
  """
  hier_label = label.split('.')
  paths =  ['.'.join(hier_label[:i+1]) for i in range(len(hier_label))]
  return paths
  

class LabelPreprocess:
  """
  Class containing helper function to process labels
  """
  
  @staticmethod
  def label_sequence(label): 
    """
    From hierarchical label as sequence get all subpaths
    
    Args:
      label (str) : hierarhical label, in format "A.B.C"
    Return
      paths (list) : hierarchical subpaths
      
    >>>LabelPreprocess.label_sequence("A.B.C")
    ["A","A.B","A.B.C"]
    """
    return get_label_sequence(label)
  
  @staticmethod
  def preprocess_labels(labels):
    """
    Get list of labels in hierarhical format.
    
    Args:
      labels (list) : list of labels
    
    Return
      hier_labels (list) : list of labels in hierarchical format
    
    >>>LabelPreprocess.preprocess_labels(["ABC","CDE","EFG"])
    ["A.B.C","C.D.E","E.F.G"]
    """
    
    def get_hier_format(label):
      return '.'.join([label[0],label[1:3],label[3]])
    
    hier_labels = [get_hier_format(l) for l in labels]
    return hier_labels
  
  @staticmethod
  def label2id(label,l2i):
    """
    Map label to index
    
    Args:
      label (str) : label
      l2i (dict) : lookup (label -> index)
    
    Return:
      index (int) : index of label
      
    """
    index = l2i.get(label)
    return index
  
  @staticmethod
  def seq_label2ids(label,seq_l2i):
    """
    Map hierarhical label sequence to ints and add EOS index.
    
    Args:
      label (str) : label
      seq_l2i (dict) : lookup (label -> index)
      
    
    >>>LabelPreprocess.seq_label2ids("A.B.C")
    [1,45,54,2]
    """
   
    seq = get_label_sequence(label)
    
    return [seq_l2i.get(l) for l in seq] + [seq_l2i.get('<EOS>')]
    
    
    
    
  
  