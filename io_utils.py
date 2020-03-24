#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:02:43 2019

@author: Samuele Garda
"""

import glob
import pickle
import logging 
import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')



class IOManager:
  """
  Class with IO helper functions
  """
  
  @staticmethod
  def files_in_folder(path,extension):
    """
    Get all the files in a folder by extension, sorted.
    
    Args:
      path (str) : system path
      extension (str) : file extensions (e.g. '.txt')
    """
    
    return sorted(glob.iglob(glob.os.path.join(path,"*.{}".format(extension))))
    
  @staticmethod
  def remove_file(path):
    """
    Delete file
    
    Args:
      path (str) : system path
    """
    
    if glob.os.path.exists(path):
      glob.os.remove(path)
      
      
  @staticmethod
  def base_name(path):
    """
    Get base name of file (no full path)
    
    Args:
      path (str) : system path
    """
    
    return glob.os.path.basename(path)
  
  @staticmethod
  def folder_name(path):
    """
    Get folder name
    
    Args:
      path (str) : system path
    """
    
    return glob.os.path.dirname(path)
    
  @staticmethod
  def check_exists(path):
    """
    Check for file existance
    
    Args:
      path (str) : system path
    """
    
    return glob.os.path.exists(path)
  
  @staticmethod
  def make_dir(path):
    """
    Create folder (and subfolders if necessary)
    
    Args:
      path (str) : system path
    """
    
    if not glob.os.path.exists(path):
      glob.os.makedirs(path)
      
  @staticmethod
  def join_paths(paths):
    """
    Create full path from list of elements
    
    Args:
      paths (list) : system paths
    """
    return glob.os.path.join(*paths)
  
  @staticmethod
  def save_pickle(item,path):
    """
    Save item in pickled format
    
    Args:
      item (whatever) : python object
      path (str) : system path
    """
    
    if not glob.os.path.exists(path):
  
      pickle.dump( item, open( str(path), 'wb'))
  
    else:
  
      raise ValueError("File {} already exists! Not overwriting".format(path))
      
  @staticmethod
  def load_pickle(path):
    """
    Load pickled object from file. 
    
    Args:
      path (str) : system path
    """
    
    if glob.os.path.exists(path):
    
      item = pickle.load(open(str(path), mode = "rb"))
  
    else:
      
      raise ValueError("File {} not found!".format(path))
      
    return item



def get_embeddings(config,indir,model_type):
  """
  Update dictionary with word and labels embeddings depending on model type.
  
  Args:
    config (dict) : hyperparameter defining layer sizes, dropout values, etc.
    indir (str) : path to folder containing TFRECORDS files
    model_type (str) : type of model (i.e. name)
  """
  config["txt_embd"] = np.float32(np.load(glob.os.path.join(indir,"txt_embd.npy")))
  
  if model_type == "ral" or model_type == "transformer":
    
    config["label_embd"] = np.float32(np.load(glob.os.path.join(indir,"seq_lab_embd.npy")))
  
  elif model_type == "leam":
    
    config["label_embd"] = np.float32(np.load(glob.os.path.join(indir,"lab_embd.npy")))
    

class ResultWriter(object):
  """
  Collect development accuracy of a model in a single file storing all hyperparameter variations
  """
  
  def __init__(self,path):
    """
    Initialize ResultWriter.
    
    Remove all elements from `params` dictionary that are not relevant as hyperparameters or
    the one that are kept fixed.    
    
    Args:
      path (str) : file containing results
    """
    
    self.path = path
    self.hps_not_to_write = ["in_dir","model_type","epochs","years","initializer_gain","eos_id",
                             "extra_decode_length","beam_size","label_smoothing","enable_metrics_in_training",
                             "learning_rate_decay_rate","learning_rate",
                             "vocab_size","optimizer_adam_beta1","optimizer_adam_beta2","optimizer_adam_epsilon",
                             "alpha","allow_ffn_pad","txt_embd","label_embd"]    
  
  def _write_header(self,conditions_name):
    """
    Write header of development result CSV file .
    
    Args:
      conditions_name (list) : list of hyperparameters of interest
    """
    
    if not glob.os.path.exists(self.path):
      with open(self.path,"w") as out_file:
        line = "config,{},dev_acc\n".format(','.join(conditions_name))
        out_file.write(line)
  
  def add_result(self,config_name,config_dict,dev_acc):
    """
    Append new development result to file.
    
    Args:
      config_name (str) : name of configuration file, i.e. specific experiment run
      config_dict (dict) : hyperparameter defining layer sizes, dropout values, etc.
      dev_acc (int) : development accuracy
    """
    
    conditions_name= sorted([k for (k,v) in config_dict.items() if k not in self.hps_not_to_write])
    
    self._write_header(conditions_name)
    
    conditions = ','.join([str(config_dict.get(hp)) for hp in conditions_name])
    
    with open(self.path, 'a') as out_file:
      line = "{},{},{}\n".format(config_name,conditions,dev_acc)
      out_file.write(line)
      
      
      
      
      
#      def h5py_dataset_iterator(g, prefix=''):
#  for key in g.keys():
#    item = g[key]
#    path = '{}/{}'.format(prefix,key)
#    if isinstance(item, h5py.Dataset): 
#      yield (path, item)
#    elif isinstance(item, h5py.Group):
#      yield from h5py_dataset_iterator(item, path)
#      
#
#if __name__ == "__main__":
#  path = sys.argv[1]
#
#  with h5py.File(path,'r') as f:
#    for path,value in h5py_dataset_iterator(f):
#      print(path)
