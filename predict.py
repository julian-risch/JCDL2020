  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:20:24 2019

@author: Samuele Garda
"""

import json
import argparse
import functools
import tensorflow as tf
from io_utils import IOManager as iom
from io_utils import get_embeddings
from tfrecords_utils import TFRecordParser
from models import model_fn

def parse_args():
  
  parser = argparse.ArgumentParser(description='Train NN models on patent classification')
  parser.add_argument('-c','--config', required=True, type = str, help='Path to configuration file')
  parser.add_argument('-t','--task', choices = ('uspto2m',), required=True, type = str, help='Path to configuration file')
  parser.add_argument('-m','--model', required=True, type = str, help='Path to model checkpoint dir')
  parser.add_argument('-o','--out', required=True, type = str, help='Path where to store predictions')
  parser.add_argument('-j','--jobs', default = 4, type = int, help='Limit to this amount of cores')

  return parser.parse_args()


def needs_flat_label(model_type):
  """
  Specify is model needs flat labels or the hierarcihcal one (seq2seq model)
  
  Args:
    model_type (str) : model type, e.g. CNN
  
  Return:
    res (bool) : whether model has sequence labels or not
  """
  
  res = model_type not in ["ral","transformer"]
  
  return res 

def get_nclasses(in_dir,model_type):
  """
  Get number of number of output classes depending on model type (flat vs seq2seq)
  
  Args:
    in_dir (str) : system path where all files for experiments are stored
    model_type (str) : model type, e.g. CNN
  
  Return:
    n_classes (int) : numbert of output classes
  """
  
  label_vocab_path = "l2i.pkl" if needs_flat_label(model_type) else "seq_l2i.pkl"
  
  n_classes = len(iom.load_pickle(iom.join_paths([in_dir,label_vocab_path])))
  
  return n_classes

def get_test_path_counts(in_dir):
  """
  Retrieve tfrecord file containing test set and the its amount of examples.
  
  Args:
    in_dir (str) : system path where all files for experiments are stored
  
  Return:
    test_path (str) : path to tfrecord file
    counts (int) : amount of example in tfrecord file
  """
  
  year_counts = json.load(open(iom.join_paths([in_dir,"year_counts.json"])))
  
  test_path = iom.join_paths([in_dir,"test.tfrecords"])
  counts = year_counts.get("test")
  
  return test_path,counts

def get_i2seq_i(in_dir):
  """
  Get a lookup mapping from flat class indices to the one in sequence format
  
  Args:
    in_dir (str) : system path where all files for experiments are stored
  
  Return:
    i2seq_i (dict) : lookup [Flat Class Index -> Seq2Seq Class Index]
  """
  
  l2i = iom.load_pickle(iom.join_paths([in_dir,"l2i.pkl"]))
  seq_l2i = iom.load_pickle(iom.join_paths([in_dir,"seq_l2i.pkl"]))
    
  i2seq_i = {v : seq_l2i.get(k) for k,v in l2i.items()}
  
  return i2seq_i


def get_prediction_fn(in_dir,model_type):
  """
  Get appropriate prediction function for model. It handles the differences in the indices
  between a flat classifier and a seq2seq one. 
  
  If the classifier is not a seq2seq model the output function  will map the predictions to the appropriate indices.
  Otherwise the function returns its input.
  
  """
  
  if needs_flat_label(model_type):
    
    # get mapping from flat index to seq2seq one
    i2seq_i = get_i2seq_i(in_dir)
    
    def pred_fn(pred,i2seq_i):
      
      # map the indices
      p = [i2seq_i.get(i) for i in pred]
      
      return p
    
    pred_fn = functools.partial(pred_fn,i2seq_i = i2seq_i)
  
  else:
    
    pred_fn = lambda x : x
  
  return pred_fn
    
 
if __name__ == "__main__":
  
  args = parse_args()
  available_cpus = args.jobs
  
  out_dir = args.out
  config_path = args.config
  task = args.task
  model_dir = args.model
  
  exp_ref_file = iom.base_name(config_path.replace(".json",".txt"))
  pred_dir = iom.join_paths([out_dir,task]) 
  iom.make_dir(pred_dir)
  pred_file = iom.join_paths([pred_dir,exp_ref_file])
  
  config = json.load(open(config_path))
  in_dir = config["in_dir"]
  model_type = config["model_type"]
    
  config["vocab_size"] = get_nclasses(in_dir,model_type)
  
  get_embeddings(config,in_dir,model_type)
  
  tfrp = TFRecordParser()
  
  test_path,test_steps_per_epochs = get_test_path_counts(in_dir)
  
  
  def test_input_fn():
    
    return tfrp.create_dataset(paths = test_path,
                               batch_size = config["batch_size"],
                               mode = "test",
                               epochs = 1,
                               max_len = config["max_len"],
                               add_sos = model_type == "ral",
                               flat_label = needs_flat_label(model_type))
    
  
  session_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=available_cpus, 
                                            inter_op_parallelism_threads=available_cpus, 
                                            allow_soft_placement=True, 
                                            device_count = {'CPU': available_cpus})
  
  run_config = tf.estimator.RunConfig(keep_checkpoint_every_n_hours=0.5, 
                                      save_checkpoints_steps= test_steps_per_epochs // 3,
                                      log_step_count_steps= test_steps_per_epochs // 3,
                                      session_config = session_config)
  
  estimator = tf.estimator.Estimator(model_fn=model_fn.model_fn,
                                     model_dir=model_dir,
                                     params=config,
                                     config=run_config)
  
  
  pred_gen = estimator.predict(input_fn=test_input_fn, yield_single_examples = True)
  
  # get function for genarating predictions
  # handle differnce in prediction indices for flat classifier and hierarchical ones
  pred_fn = get_prediction_fn(in_dir,model_type)
  
  with open(pred_file,"w+") as outfile:
    for i,pred in enumerate(pred_gen):
      pred = pred_fn(pred)
      str_pred = "{}\n".format(" ".join(str(x) for x in pred))
      outfile.write(str_pred)
