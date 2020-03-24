#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:03:54 2019

@author: Samuele Garda
"""

import json
import argparse
import tensorflow as tf
from io_utils import IOManager as iom
from io_utils import ResultWriter,get_embeddings
from tfrecords_utils import TFRecordParser
from models import model_fn 

def parse_args():
  
  parser = argparse.ArgumentParser(description='Train trasformer on patent classification')
  parser.add_argument('-c','--config', required=True, type = str, help='Path to configuration file')
  parser.add_argument('-t','--task', choices = ('uspto2m',), required=True, type = str, help='Path to configuration file')
  parser.add_argument('-r','--results', required=True, type = str, help='Dir where DEVELOPMENT results are stored')
  parser.add_argument('-m','--models', required=True, type = str, help='Dir where models are stored')
  parser.add_argument('-j','--jobs', default = 4, type = int, help='Limit to this amount of cores')

  return parser.parse_args()

def get_train_dev_paths_counts(in_dir,batch_size,names = "all"):
  
  year_counts = json.load(open(iom.join_paths([in_dir,"year_counts.json"])))
  
  year_counts.pop("test")
   
  infiles = iom.files_in_folder(in_dir,extension = "tfrecords")
  
  in_dict = {iom.base_name(f).replace(".tfrecords","") : f  for f in infiles}
  
  in_dict.pop("test")
  
  dev_path = in_dict.pop("dev")
  
  dev_step_per_epoch = year_counts.get("dev") // batch_size
  
  train_years = list(in_dict.keys()) if names == "all" else [v for (k,v) in in_dict.items() if k in names]
    
  train_path = [in_dict.get(n) for n in train_years] 
  
  train_steps_per_epoch = sum([year_counts.get(n) for n in train_years]) // batch_size
  
  return train_path,train_steps_per_epoch,dev_path,dev_step_per_epoch


def get_test_path_counts(in_dir):
  
  year_counts = json.load(open(iom.join_paths([in_dir,"year_counts.json"])))
  
  test_path = iom.join_paths([in_dir,"test.tfrecords"])
  
  return test_path,year_counts.get("test")

def needs_flat_label(model_type):
  
  return model_type not in ["ral","transformer"]


def get_nclasses(in_dir,model_type):
  
  label_vocab_path = "l2i.pkl" if needs_flat_label(model_type) else "seq_l2i.pkl"
  
  n_classes = len(iom.load_pickle(iom.join_paths([in_dir,label_vocab_path])))
  
  return n_classes


def get_transformer_warmup_steps(train_counts,epochs):
  
  warmup = int((train_counts * epochs * 4000)/100000)
  
  return warmup
  
if __name__ == "__main__":
  
  args = parse_args()
  
  available_cpus = args.jobs
  
  config_path = args.config
  task = args.task

  res_dir = iom.join_paths([args.results,task])
  exp_ref_name = iom.base_name(config_path.replace(".json",""))
  
  model_dir = iom.join_paths([args.models,exp_ref_name])
  
  iom.make_dir(res_dir)
  iom.make_dir(model_dir)
  
   
  config = json.load(open(config_path))
  in_dir = config["in_dir"]
  model_type = config["model_type"]  
  res_file = iom.join_paths([res_dir,"{}.csv".format(model_type)])
  
  train_path,train_steps_per_epoch,dev_path,dev_steps_per_epoch = get_train_dev_paths_counts(in_dir,
                                                                                               batch_size = config["batch_size"],
                                                                                               names = config["years"])
  
  config["vocab_size"] = get_nclasses(in_dir,model_type)
  get_embeddings(config,in_dir,model_type)
  config["learning_rate_warmup_steps"] =  get_transformer_warmup_steps(train_counts = train_steps_per_epoch,epochs = config["epochs"])
  
  tfrp = TFRecordParser()
      
  session_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=available_cpus, 
                                            inter_op_parallelism_threads=available_cpus, 
                                            allow_soft_placement=True, 
                                            device_count = {'CPU': available_cpus})
    
  run_config = tf.estimator.RunConfig(keep_checkpoint_every_n_hours=0.5, 
                                      save_checkpoints_steps= train_steps_per_epoch // 3,
                                      log_step_count_steps= 100, #train_steps_per_epoch // 3,
                                      session_config = session_config)
    
  estimator = tf.estimator.Estimator(model_fn=model_fn.model_fn,
                                     model_dir=model_dir,
                                     params=config,
                                     config=run_config)
        

    
  def train_input_fn():
    
    return tfrp.create_dataset(paths = train_path,
                               batch_size = config["batch_size"],
                               mode = "train",
                               epochs = config["epochs"],
                               max_len = config["max_len"],
                               add_sos = model_type == "ral",
                               flat_label = needs_flat_label(model_type),
                               npc = available_cpus)
    
    
      
  dataset = train_input_fn()
  
  def eval_input_fn():
    
    return tfrp.create_dataset(paths = dev_path,
                               batch_size = config["batch_size"],
                               mode = "train",
                               epochs = 1,
                               max_len = config["max_len"],
                               add_sos = model_type == "ral",
                               flat_label = needs_flat_label(model_type),
                               npc = available_cpus)
    
    

  res_writer = ResultWriter(path = res_file)
    
  estimator.train(input_fn = train_input_fn)
  
  eval_results = estimator.evaluate(input_fn = eval_input_fn)
  
  res_writer.add_result(config_name = exp_ref_name,
                        config_dict = config,
                        dev_acc = eval_results.get("metrics/accuracy"))

    