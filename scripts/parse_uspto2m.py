#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:41:49 2019

@author: Samuele Garda
"""

import logging
import argparse
import pandas as pd
import smart_open 
from collections import Counter
from gensim.utils import to_utf8
from io_utils import IOManager as iom
from pp_utils import TextPreprocess as tp
from pp_utils import LabelPreprocess as lp

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')


DEV_SIZE_USPTO2M = 49900

def parse_args():
  
  parser = argparse.ArgumentParser(description='Create pre-tokenized year files and vocabulary of USPTO2M')
  parser.add_argument('-d','--dir', required=True, type = str, help='Folder where corpus in JSON format is stored')
  parser.add_argument('-o','--out', required=True, type = str, help='Folder where to store parsed corpus')
  
  return parser.parse_args()


def update_word_freq(words,freqs = None):
  
  if freqs is not None:
    
    for word in words:
        
      freqs[word] += 1
      
      
def update_label_freq(labels,freqs,seq_freqs):
  
  for label in labels:
    
    freqs[label] += 1
    
    seq = lp.label_sequence(label)
    
    for t in seq:
            
      seq_freqs[t] += 1

def get_instance_byte_update_word_freq(row,word_freqs,label_freqs,seq_label_freqs):
  
  abstract = tp.preprocess_string(row["Abstract"])
  update_word_freq(abstract,word_freqs)
  abstract = ' '.join(abstract)
  
  labels = lp.preprocess_labels(row["Subclass_labels"])
  update_label_freq(labels,
                    freqs = label_freqs,
                    seq_freqs = seq_label_freqs)
  labels = ' '.join(labels)
  
  line = to_utf8("{}\t{}\n".format(abstract,labels)) if labels is not None else None
  
  return line


def write_compressed_file(df,outname,word_freqs,label_freqs,seq_label_freqs):
  
  logger.info("Start creating file {}".format(outname))
  with smart_open.open(outfile_name,'wb') as outfile:
    for row_idx,row in df.iterrows():
      line = get_instance_byte_update_word_freq(row = row,
                                                word_freqs = word_freqs,
                                                label_freqs = label_freqs,
                                                seq_label_freqs  = seq_label_freqs)
      outfile.write(line)
      if (row_idx%10000)==0:
        logger.info("Parsed {} instances".format(row_idx))
      
    logger.info("Finished creating file {}".format(outname))
  

if __name__ == "__main__":
  
  
  args = parse_args()
  
  in_dir = args.dir
  out_dir = args.out
  
  iom.make_dir(out_dir)
  
  infiles = iom.files_in_folder(in_dir,extension = "json")
  word_freqs_path = iom.join_paths([out_dir,"w_freqs.pkl"])
  label_freqs_path = iom.join_paths([out_dir,"l_freqs.pkl"])
  seq_label_freqs_path = iom.join_paths([out_dir,"ls_freqs.pkl"])
  
  word_freqs = Counter()
  label_freqs = Counter()
  seq_label_freqs = Counter()
  
  for infile in infiles:
    outfile_base = iom.base_name(infile).replace("_USPTO.json","")
    outfile_base = outfile_base if outfile_base != "2015" else "test"
    if outfile_base == "2014":
      reader = pd.read_json(infile)
      
      train_idx = len(reader) - DEV_SIZE_USPTO2M
      logger.info("Using first {} instances of year 2014 for training".format(train_idx))
      train = reader[0:train_idx]

      outfile_name = iom.join_paths([out_dir,outfile_base + ".txt.gz"])
      write_compressed_file(train,outname = outfile_name,
                            word_freqs = word_freqs,label_freqs =label_freqs,seq_label_freqs = seq_label_freqs)
      
      dev = reader[train_idx:]
      logger.info("Using last {} instances as development set".format(dev.shape[0]))
      outfile_name = iom.join_paths([out_dir,"dev.txt.gz"])
      write_compressed_file(dev,outname = outfile_name ,
                            word_freqs = word_freqs,label_freqs =label_freqs,seq_label_freqs = seq_label_freqs)
    else:
      outfile_name = iom.join_paths([out_dir,outfile_base + ".txt.gz"])
  
      reader = pd.read_json(infile)
      write_compressed_file(reader,outname = outfile_name ,
                            word_freqs = word_freqs,label_freqs =label_freqs,seq_label_freqs = seq_label_freqs)
      
  iom.save_pickle(word_freqs,word_freqs_path)
  logger.info("Saved vocabulary of size at : `{}` of size {}".format(word_freqs_path,len(word_freqs)))
  iom.save_pickle(label_freqs,label_freqs_path)
  logger.info("Saved single label vocabulary of size at : `{}` of size {}".format(label_freqs_path,len(label_freqs)))
  iom.save_pickle(seq_label_freqs,seq_label_freqs_path)
  logger.info("Saved sequence labels vocabulary  of size at : `{}` of size {}".format(seq_label_freqs_path,len(seq_label_freqs)))
        
