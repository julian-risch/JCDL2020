#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:30:39 2019

@author: Samuele Garda

"""

import logging
import argparse
import numpy as np
from scipy import sparse as sps
from subprocess import Popen, PIPE
from io_utils import IOManager as iom
from pp_utils import get_label_sequence

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')



def parse_args():
  """
  Read command line arguments.
  """
  
  parser = argparse.ArgumentParser(description='Evaluate predictions')
  parser.add_argument('--dir', required = True, help = "TFRECORD FOLDER")
  parser.add_argument('--pred', required = True, help = 'Path prediction file')
  parser.add_argument('--hemkit', required = True, help = 'Path to HEMkit bin file.')
  parser.add_argument('--max-dist', default = "5", help = 'HEMkit : above this threshold all nodes will be considered to have a common ancestor.')
  parser.add_argument('--pb-max-err', default = "4", help = 'HEMkit : maximum error with which pair-based measures penalize nodes that were matched with default.')
  
  args = parser.parse_args()
  
  return args


  
class Evaluation(object):
  """
  Run evaluation of a model prediction againts test set.
  Compute patent specific measures, i.e. :
    - Top Prediction (accuracy)
    - Three Guesses (top 3 predictions vs top true class)
    - All Guesses (top prediction vs top 3 true classes)
  
  and hierarchical evaluation implemented as https://arxiv.org/abs/1306.6802, i.e.:
    - Lowest Common Ancestor Precision
    - Lowest Common Ancestor Recall
    - Lowest Common Ancestor F1 score
  
  with HEMkit software found at https://github.com/BioASQ/Evaluation-Measures/tree/master/hierarchical
  """
  
  def create_oneline_hierarchy_file(self,hl2is_path,cat_hier):
    """
    Needed for HEMkit : file containing all hierarchical relations as pairs.
    File is created only it does not exists
    
    Args:
      hl2is_path (str) : path to pickled dict containing all hierarchical labels
      cat_hier (str) : path where to save output file
    """
  
    if not iom.check_exists(cat_hier):
      hl2is = iom.load_pickle(hl2is_path)
      
      rel_cache = set()
      
      with open(cat_hier,'w+') as out_file:
        for l,i in hl2is.items():
          paths = get_label_sequence(l)
          for idx in range(len(paths)-1):
            parent = hl2is.get(paths[idx])
            child = hl2is.get(paths[idx+1])
            rel = (parent,child)
            if rel not in rel_cache:
              rel_cache.add(rel)
              out_file.write("{} {}\n".format(parent,child))
      
    else:
      logger.info("File `{}` already exists! Not overwriting!".format(cat_hier))

  def count_elements(self,infile):
    """
    Reads file and return number of new lines.
    """
    
    return open(infile).read().count("\n")

  def length_sanity_check(self,true_size,pred_size):
    """
    Assertion two numbers are equal.
    """
    
    assert true_size == pred_size, "# labels != # predictions! Found : {} - {}".format(true_size,pred_size)

  def build_matrix(self,file,size):
    """
    Create sparse matrix of shape (num_examples,3) from file.
    
    Args:
      file (str) : path to file containing top 3 prediction per row 
      size (int) : number of elements (per line) contained in file
    """
    
    matrix = sps.lil_matrix((size,3), dtype = int)
    
    with open(file) as infile:
      for idx,line in enumerate(infile):
        values = np.fromstring(line, dtype = int, sep = " ")
        if len(values) < 3:
          values = np.pad(values, (0, 3 - len(values)), mode = 'constant', constant_values = 0)
        matrix[idx,0] = values[0]
        matrix[idx,1] = values[1]
        matrix[idx,2] = values[2]
    
    return matrix.tocsr()
  
  def __get_inverted_bool_matrix(self,matrix):
    """
    Helper function to create sparse matrix containing all 1.
    
    Args:
      matrix (scipy.sparse) : matrix from which to derive shape
    
    Return:
      invert_bool (scipy.sparse) : out matrix
    """
    
    invert_bool_lil = sps.lil_matrix((matrix.shape[0],1))
    invert_bool_lil[:,:] = 1
    invert_bool = invert_bool_lil.tocsr()
    
    return invert_bool
  
  def accuracy(self,y_true,y_pred):
    """
    Compute accuracy over column row of matrix.
    
    Args:
      y_true (scipy.sparse) : matrix (num_examples,3)
      y_pred (scipy.sparse) : matrix (num_examples,3)
    
    Return
      score (int) : metric

    """
    
    acc = y_true[:,0].todense() == y_pred[:,0].todense()
    
    return np.average(acc)
    
  def three_guesses(self,y_true,y_pred,invert_bool):
    """
    Compute Three Guesses metric : top 3 predictions vs top true class
    
    Args:
      y_true (scipy.sparse) : matrix (num_examples,3)
      y_pred (scipy.sparse) : matrix (num_examples,3)
    
    Return
      score (int) : metric
    """
  
    first_guess = invert_bool - (y_pred[:,0] != y_true[:,0]) 
    second_guess = invert_bool - (y_pred[:,1] != y_true[:,0])
    third_guess = invert_bool - (y_pred[:,2] != y_true[:,0])
    
    full = sps.hstack([first_guess,second_guess,third_guess])
  
    scores = full.sum(axis = 1)
        
    score = scores.mean()
    
    return score

  def all_guesses(self,y_true,y_pred,invert_bool):
    """
    Compute All Guesses : top prediction vs top 3 true classes
    
    Args:
      y_true (scipy.sparse) : matrix (num_examples,3)
      y_pred (scipy.sparse) : matrix (num_examples,3)
    
    Return
      score (int) : metric

    """
    
    first_guess = invert_bool - (y_true[:,0] != y_pred[:,0]) 
    second_guess = invert_bool - (y_true[:,1] != y_pred[:,0])
    third_guess = invert_bool - (y_true[:,2] != y_pred[:,0])
    
    full = sps.hstack([first_guess,second_guess,third_guess])
  
    scores = full.sum(axis = 1)
        
    score = scores.mean()
    
    return score
  
  def run_patent_eval(self,file_true,file_pred):
    """
    Run all patent metrics.
    
    Args:
      file_true (str) : path containing true classes: each line is in the form (example,top_3_classes)
      file_pred (str) : path containing pred classes: each line is in the form (example,top_3_classes)
    
    Return:
      tp (Top Prediction), tg (Three Guesses), ag (All Guesses)
    """
    
    size_true = self.count_elements(file_true)
    size_pred = self.count_elements(file_pred)
    
    self.length_sanity_check(size_true,size_pred)
    
    y_true = self.build_matrix(file_true,size_true)
    y_pred = self.build_matrix(file_pred,size_pred)
    
    inv_bool = self.__get_inverted_bool_matrix(y_pred)
    
    tp = self.accuracy(y_true,y_pred)
    tg = self.three_guesses(y_true,y_pred,inv_bool)
    ag = self.all_guesses(y_true,y_pred,inv_bool)
    
    return tp,tg,ag
  
  def run_hier_eval(self,file_true,file_pred,hemkit,label_hier,max_dist,pb_max_err):
    """
    Run hierarchical evaluation metrics with HEMkit.
    
    Args:
      file_true (str) : path containing true classes: each line is in the form (example,top_3_classes)
      file_pred (str) : path containing pred classes: each line is in the form (example,top_3_classes)
      hemkit (str) : path to HEMkit bin file
      label_hier (str) : path to hierarchi file
      max_dist (int) : above this threshold all nodes will be considered to have a common ancestor
      pb_max_err (int) : maximum error with which pair-based measures penalize nodes that were matched with default
    
    Return:
      LCA = Lowest Common Ancestor
      lcap (LCA Precision), lcar (LCA Recall), lcar (LCA F1)
    """
    
    def get_hemkit_result(res):
      hemkit_res = res[0].decode().split()
      return hemkit_res
    
    process = Popen([hemkit, label_hier, file_true, file_pred, max_dist, pb_max_err], 
                    stdout=PIPE, stderr=PIPE, stdin = PIPE)
    
    hP,hR,hF,lcap,lcar,lcaf1 = get_hemkit_result(process.communicate())
    
    return lcap,lcar,lcaf1
    
    
 
if __name__ == "__main__":
  
  args = parse_args()
  
  in_dir = args.dir
  seq_l2i_file = iom.join_paths([in_dir,"seq_l2i.pkl"])
  label_hierarchy_file = iom.join_paths([in_dir,"label.hier"])
  true_file = iom.join_paths([in_dir,"test.txt"])
  
  pred_file = args.pred
  hemkit = args.hemkit
  max_dist = args.max_dist
  pb_max_err = args.pb_max_err
  
  evaluation = Evaluation()
  
  evaluation.create_oneline_hierarchy_file(seq_l2i_file,
                                           label_hierarchy_file)
  
  tp,tg,ag = evaluation.run_patent_eval(true_file,pred_file)
  
  print("TP : {} - TG : {} - AG : {}".format(tp,tg,ag))
  
  lcap,lcar,lcaf1 = evaluation.run_hier_eval(true_file,pred_file,
                                             hemkit,label_hierarchy_file,max_dist,pb_max_err)
  
  print("LCA | P : {} - R : {} - F1 : {}".format(lcap,lcar,lcaf1))
  
  
  
  
  
  
  
  
    
