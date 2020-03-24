#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:34:30 2019

@author: Samuele Garda
"""

import argparse
import logging
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText
from io_utils import IOManager as iom
from pp_utils import LabelPreprocess as lp
from pp_utils import TextPreprocess as tp

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')

def parse_args():
  
  parser = argparse.ArgumentParser(description='Create embeddings matrices')
  parser.add_argument('-p','--path', required=True, type = str, help='Path to embeddings file')
  parser.add_argument('-m','--mapping', required=True, type = str, help='Path to IPC mapping')
  parser.add_argument('-o','--out', type = str, help='FOLDER TO TFRECORDS')
  
  return parser.parse_args()


def load_mapping(mapping):
  
  def _value_is_list(mapping):
  
    return isinstance(next(iter(mapping.values())).get('hier'),list)
  
  m = iom.load_pickle(mapping)
  
  if _value_is_list(m):
    new = {el : value.get('desc') for (key,value) in m.items() for el in value.get('hier')}
  else:
    new = {value.get('hier') : value.get('desc') for (key,value) in m.items()}
    
  del m
  
  return new


def load_fasttext(path):
  
  model =  FastText.load_fasttext_format(path)
  
  model.init_sims(replace = True)
  
  return model


def get_base_matrix(lookup,model):
  
  embedding_matrix = np.random.uniform(low = -0.001, 
                                       high = 0.001,  
                                       size = (len(lookup)+1, model.vector_size))

  return embedding_matrix


def create_text_embeddings(w2i,model,outpath):
  
  logger.info("Creating word embedding matrix")
  
  embedding_matrix = np.random.uniform(low = -0.001, 
                                       high = 0.001,  
                                       size = (len(w2i)+1, model.vector_size))
  
  not_found = 0
  
  for word,idx in w2i.items():
    try:
      embedding_matrix[idx] = model[word]
    except KeyError:
      not_found += 1
  
  np.save(outpath,embedding_matrix)
  
  if not_found:
    logger.info("TEXT : Word vectors not found : {}".format(not_found))
  logger.info("TEXT : Saved word embedding matrix with shape {} at `{}`".format(embedding_matrix.shape,outpath))

def create_label_embeddings(l2i,model,mapping,outpath,name = "LABELS"):
  
  num_labels = len(l2i)+1 if name == "LABELS" else len(l2i)+2
  
  embedding_matrix = np.random.uniform(low = -0.001, 
                                       high = 0.001,  
                                       size = (num_labels, model.vector_size))
  
  not_found = 0
  not_in_mapping = 0
  
  for label,idx in l2i.items():
    
    if label in mapping:
      
      paths = lp.label_sequence(label)
      
      desc = [mapping.get(p) for p in paths]
              
      desc = set(tp.preprocess_string(' '.join([w for sublist in desc for w in sublist])))
                  
      vectors = []
      
      for w in desc:
        try:
          vec = model[w]
          vectors.append(vec)
        except KeyError:
          pass
      
      if not len(vectors):
        not_found += 1
        continue
      
    
      vec = np.mean(vectors, axis = 0)
      
      embedding_matrix[idx] = vec
      
    else:
      not_in_mapping += 1 
      continue
    
  np.save(outpath,embedding_matrix)
  
  logger.info("{} : Labels with randomly init vector : {}".format(name,not_found))
  logger.info("{} : Labels not in mapping : {}".format(name,not_in_mapping))
  logger.info("{} : Saved label embedding matrix with shape {} at `{}`".format(name,embedding_matrix.shape,outpath))



if __name__ == "__main__":
  
  args = parse_args()
  
  model_path = args.path
  out_dir = args.out
  mapping_path = args.mapping
  
  
  model = KeyedVectors.load_word2vec_format(model_path,limit = 10000)
  model.init_sims(replace = True)
  
  
#  model = load_fasttext(model_path)
  mapping = load_mapping(mapping = mapping_path)
  
  w2i = iom.load_pickle(iom.join_paths([out_dir,"w2i.pkl"]))
  l2i = iom.load_pickle(iom.join_paths([out_dir,"l2i.pkl"]))
  seq_l2i = iom.load_pickle(iom.join_paths([out_dir,"seq_l2i.pkl"]))
  
  create_text_embeddings(w2i = w2i,
                         model = model,
                         outpath = iom.join_paths([out_dir,"txt_embd.npy"]))
  
  create_label_embeddings(l2i = l2i,
                          model = model,
                          mapping = mapping,
                          outpath = iom.join_paths([out_dir,"lab_embd.npy"])
                         )
  
  create_label_embeddings(l2i = seq_l2i,
                          model = model,
                          mapping = mapping,
                          outpath = iom.join_paths([out_dir,"seq_lab_embd.npy"]),
                          name = "HIER-LABEL"
                         )
  
  
  
  
  
  
