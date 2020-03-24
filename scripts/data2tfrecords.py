#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:47:01 2019

@author: Samuele Garda
"""
import json
import logging
import argparse
import smart_open
from tfrecords_utils import TFRecordWriter
from io_utils import IOManager as iom
from pp_utils import TextPreprocess as tp
from pp_utils import LabelPreprocess as lp


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')

def parse_args():
  
  parser = argparse.ArgumentParser(description='Create train, dev and test tfrecords files')
  parser.add_argument('-d','--dir', required=True, type = str, help='Dir where parsed corpus is stored')
  parser.add_argument('-t','--task', choices = ('uspto2m',), required=True, type = str, help ='Dataset to be parsed: `same_side` or transfer nli  `nli`')
  parser.add_argument('-m','--max-freq', default = None, type = int, help='Drop words that are less frequent than this')
  parser.add_argument('-o','--out', required=True, type = str, help='Dir where to store tfrecords')
  
  return parser.parse_args()

def get_vocabs(w2freqs,l2freqs,seq_l2freqs,max_freq):
  
  vocab = w2freqs.most_common(max_freq)
  
  
  logger.info("Loaded word 2 index lookup")
  
  if max_freq:
    logger.info("Using {} most common words : {}...".format(max_freq,vocab[:10]))
    
  w2i = {k : i for i,(k,v) in enumerate(vocab,start = 2)}
  w2i["<S>"] = 1
    
  
  l2i = {k : i for i,(k,v) in enumerate(l2freqs.items())}
  
  logger.info("Loaded label 2 index lookup")
  
  logger.info("Loaded : {}".format(len(w2i)))
  
  seq_l2i = {k : i for i,(k,v) in enumerate(seq_l2freqs.items(),start = 3)}
  seq_l2i["<EOS>"] = 2
  
  logger.info("Loaded label as sequences 2 index lookup ")
  
  return w2i,l2i,seq_l2i


def dump_year_counts(path,counts):
  
  with open(path,"w") as out_file:
    json.dump(counts,out_file,indent = 1)
  
  logger.info("Saved number of instances per year at `{}`".format(path))
  
  
def create_hemkit_true_labels_file(test_file,out_dir,seq_l2i):
  
  pred_file = iom.join_paths([out_dir,"test.txt"])
  
  with smart_open.open(test_file) as infile, open(pred_file,"w") as out_file:
    
    for line in infile:
      
      text,labels = line.strip("\n").split("\t")
      
      labels = labels.split()[:3]
      
      labels = " ".join(str(seq_l2i.get(l)) for l in labels)
      
      out_file.write("{}\n".format(labels))
  
  


if __name__ == "__main__":
  
  args = parse_args()
  
  in_dir = args.dir
  max_freq = args.max_freq
  task = args.task
  out_dir = args.out
  
  iom.make_dir(out_dir)
  
  infiles = iom.files_in_folder(in_dir,extension = "txt.gz")
  
  w2i,l2i,seq_l2i = get_vocabs(w2freqs = iom.load_pickle(iom.join_paths([in_dir,"w_freqs.pkl"])),
                               l2freqs = iom.load_pickle(iom.join_paths([in_dir,"l_freqs.pkl"])),
                               seq_l2freqs = iom.load_pickle(iom.join_paths([in_dir,"ls_freqs.pkl"])),
                               max_freq = max_freq)
  
  tfrw = TFRecordWriter(tp = tp, lp = lp)
  year_counts_path = iom.join_paths([out_dir,"year_counts.json"])
  w2i_path = iom.join_paths([out_dir,"w2i.pkl"])
  l2i_path = iom.join_paths([out_dir,"l2i.pkl"])
  seq_l2i_path = iom.join_paths([out_dir,"seq_l2i.pkl"])
  
  year2n_examples = {}
  
  for infile in infiles:
    
    logger.info("Start processing {}".format(infile))
    
    base_name = iom.base_name(infile).replace('.txt.gz','')
    out_name = iom.join_paths([out_dir,base_name + '.tfrecords'])
    
    n_examples  = tfrw.write_file_tfrecord(in_file = infile,
                                           out_file = out_name,
                                           w2i = w2i,
                                           l2i = l2i,
                                           seq_l2i = seq_l2i)
    
    logger.info("Finished processing {}".format(infile))
    
    year2n_examples["{}".format(base_name)] = n_examples
  
  
  dump_year_counts(year_counts_path,year2n_examples)
  
  logger.info("Completed writing data set in tfrecords format")
    
  logger.info("Saving Lookups")
    
  iom.save_pickle(w2i,w2i_path)
    
  iom.save_pickle(l2i,l2i_path)
  
  iom.save_pickle(seq_l2i,seq_l2i_path)
  
  create_hemkit_true_labels_file(test_file = [file for file in infiles if "test" in file][0],
                                 out_dir = out_dir)
  
  
    
    
  
    
  
  
  


  
  
  
  
  
  
