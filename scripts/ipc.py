#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 16:15:03 2018

@author: Samuele Garda
"""

import re
import logging
import argparse
import json
from collections import defaultdict
from lxml import etree
from io_utils import IOManager as iom

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO') 

def parse_arguments():
  """
  Parse arguments.
  """
  
  parser = argparse.ArgumentParser(description='Parse original IPC labeling scheme (XML file) to create label mapping.')
  parser.add_argument('-i', '--ipc', required=True, help='Path to ìpc scheme in XML format')
  parser.add_argument('-p', '--parsed', required=True, help='Path where to store parsed IPC')
  parser.add_argument('-m', '--mapping', required=True, help='Path where to store IPC mapping : Base  label -> Hierarchical label and description')
  
  return parser.parse_args()


RM_CHAR = re.compile('[A-Z]\.')

CATEGORIES = {"A": "HUMAN NECESSITIES",
              "B": "PERFORMING OPERATIONS; TRANSPORTING",
              "C": "CHEMISTRY; METALLURGY",
              "D": "TEXTILES; PAPER",
              "E": "FIXED CONSTRUCTIONS",
              "F": "MECHANICAL ENGINEERING; LIGHTING; HEATING; WEAPONS; BLASTING",
              "G": "PHYSICS",
              "H": "ELECTRICITY"}


class IPCStats:
  """
  Wrapper class to get general stats about IPC.
  """
  
  @staticmethod
  def get_ipc_classes_by_level(ipc_mapping):
  
    classes = defaultdict(set)
        
    for basename in ipc_mapping:
      tree_num = ipc_mapping.get(basename).get('hier')
   
      tree_num = tree_num.split('.')
      len_tree = len(tree_num)
          
      for i in range(len_tree):
        classes[i].add(tree_num[i])
    
    return classes
  
  @staticmethod
  def get_ipc_labels_by_level(ipc_mapping):
  
    classes = defaultdict(set)
    
    for basename in ipc_mapping:
      tree_num = ipc_mapping.get(basename).get('hier')
    
      tree_num = tree_num.split('.')
      
      len_tree = len(tree_num) - 1
      
      classes[len_tree].add(basename)
      
    return classes
  
  @staticmethod
  def get_ipc_label_desc_by_level(ipc_mapping):
  
    labels_desc_mapping = defaultdict(list)
    
    for basename in ipc_mapping:
       
       tree_num = ipc_mapping.get(basename).get('hier')
       labels_desc = ipc_mapping.get(basename).get('desc') 
       len_tree = len(tree_num.split('.')) - 1 
          
       labels_desc_mapping[len_tree].append([word for v in labels_desc for word in v.split()])
          
    return labels_desc_mapping
      

def ipc_tree_num_to_mesh_style(tree_num):
  """
  Get IPC label in hierarchical form. Same style as MeSH. 
  The label must come from IPC XML file and must have at least a sub-group.
  
  >>> ipc_tree_num_to_mesh_style('A01B0063112000')
  'A.01.B.63.2.1.111.112'
  
  :params:
    tree_num (str) : base label
  :return:
    tree_num_mesh_syle (str) : label in hierarchical form
  """
  
  assert len(tree_num) == 14, logger.error("IPC tree number must be full path! But {} is {}!".format(tree_num,len(tree_num)))
  
  tree_num_mesh_syle = ""
  
  # category
  tree_num_mesh_syle += "{}.".format(tree_num[0])
  # class
  tree_num_mesh_syle += "{}.".format(tree_num[1:3])
  #subclass
  tree_num_mesh_syle += "{}.".format(tree_num[3])
  # group
  tree_num_mesh_syle += "{}".format(tree_num[6:8])
  
  firs_code_subgroup = 1
  # subgroup : remove extra zeros 
  for idx,code in enumerate(tree_num[8:]):
    if code == '0':
      if idx == 0 or idx == 1:
        pass
#        tree_num_mesh_syle += code
#      else:
#        pass
    else:
      if firs_code_subgroup:
        tree_num_mesh_syle += ".{}".format(code)
        firs_code_subgroup = 0
      else:
        tree_num_mesh_syle += code

  
  return tree_num_mesh_syle


def icp_short_tree_num_to_mesh_style(tree_num):
  """
  Get IPC label in hierarchical form. Same style as MeSH.
  The label must come from IPC XML file and must have maximum a subclass.
  
  >>>icp_short_tree_num_to_mesh_style("HO9Z")
  'H.09.Z'
  
  :params:
    tree_num (str) : base label
  :return:
    tree_num_mesh_syle (str) : label in hierarchical form
  """
  
  tree_num_mesh_style = ""
  
  if len(tree_num) == 1:
    tree_num_mesh_style = tree_num
  
  if len(tree_num) == 3:
    tree_num_mesh_style += "{}.".format(tree_num[0])
    tree_num_mesh_style += "{}".format(tree_num[1:3])
    
  if len(tree_num) == 4:
    tree_num_mesh_style += "{}.".format(tree_num[0])
    tree_num_mesh_style += "{}.".format(tree_num[1:3])
    tree_num_mesh_style += "{}".format(tree_num[-1])

    
  return tree_num_mesh_style
    


def get_child_full_path(parent_tree_num,child_tree_num):
  """
  Help function to generate full hierarchical path of a label given its parent.
  Both inputs labels must be in hierarchical form.
  
  >>> get_child_full_path(''A.01.B.01.6'','A.01.B.01.14')
  'A.01.B.01.6.14'
  
  
  :params:
    parent_tree_num (str) : parent label in hierarchical form
    child_tree_num (str) : child label in hierarchical form
  :return:
    child_full_path (str) : child full hierarchical form
  """
  
  parent_nodes = parent_tree_num.split('.')
  child_nodes = child_tree_num.split('.')

  child_level = list(set(child_nodes) - set(parent_nodes))
      
  child_full_path = '.'.join(parent_nodes + child_level)
  
  return child_full_path


def get_mapping_flat_to_hier_from_parsed_ipc_scheme(ipc_scheme_parsed):
  """
  Parse the parsed version (JSON) of IPC labeling system and create a mapping from a base label to its hierarchical version.
  
  >>> mapping = get_mapping_flat_to_hier_from_parsed_ipc_scheme('parsed.json')
  >>> mapping.get('A01B0001140000')
  'A.01.B.01.6.14'
  
  :params:
    ipc_scheme_parsed (str) : path to parsed IPC
  :return:
    mapping (dict) : mapping from a base label to its hierarchical version
  """
  
  logger.info("Parsing `{}` for creating mapping : base label -> hierarchical".format(ipc_scheme_parsed))
    
  mapping = {}
  
  with open(ipc_scheme_parsed) as infile:
    for idx,line in enumerate(infile):
              
      line = json.loads(line)
      
      parent = line.get('node')
      children = line.get('children')
      
      if len(parent) < 14:
        
        mapping[parent] = icp_short_tree_num_to_mesh_style(parent)
        
      else:
        
        if parent in mapping:
          parent_mesh_style = mapping.get(parent)
        else:
          parent_mesh_style = ipc_tree_num_to_mesh_style(parent)
          mapping[parent] = parent_mesh_style
        
        if children:
                                
          for child in children:
                            
            child_mesh_style = ipc_tree_num_to_mesh_style(child)
              
            child_mesh_style_full = get_child_full_path(parent_mesh_style, child_mesh_style)
              
            mapping[child]= child_mesh_style_full
            mapping[child] = child_mesh_style_full
  
  logger.info("Completed creation of mapping : label -> hierarchical label.")
  
  return mapping


def get_mapping_flat_to_desc_from_parsed_ipc_scheme(ipc_scheme_parsed):
  """
  Parse the parsed version (JSON) of IPC labeling system and create a mapping from a base label to its description.
  
  >>> mapping = get_mapping_flat_to_desc_from_parsed_ipc_scheme('parsed.json')
  >>> mapping.get('A01B0001140000')
  ['with teeth only']
  
  :params:
    ipc_scheme_parsed (str) : path to parsed IPC
  :return:
    mapping (dict) : mapping from a base label to its description
    
  """
  
  logger.info("Parsing `{}` for creating mapping : base label -> label description".format(ipc_scheme_parsed))
  
  mapping = {}
  
  with open(ipc_scheme_parsed) as infile:
    for idx,line in enumerate(infile):
      line = json.loads(line)
      
      node = line.get('node')
      desc = line.get('desc')
      
      if not node in mapping:
        mapping[node] = desc
      
  logger.info("Completed creation of mapping : label -> description.")
  
  return mapping


def merge_mappings(flat_to_hier,flat_to_desc):
  """
  Merge (1) mappping : label -> description and (2) label -> hierarchical label.
 
  >>> mapping = merge_mappings(flat_to_hier,flat_to_desc)
  >>> mapping.get('A01B0001140000')
  {'desc': ['with teeth only'], 'hier': 'A.01.B.01.6.14'}
  
  :params:
    flat_to_hier (dict) :mapping from a base label to its hierarchical version
    flat_to_desc (dict) : mapping from a base label to its description
    
  :return:
    mapping (dict of dict): keys are flat labels. Values are a dict with keys : `hier` for hierarchical version  and `desc` for label description
  """
  
  mapping = defaultdict(dict)
  
  for flat_label in flat_to_hier:
    
    mapping[flat_label]['hier'] = flat_to_hier.get(flat_label)
    mapping[flat_label]['desc'] = flat_to_desc.get(flat_label)
    
  return mapping
  

def rec_parse_ipc_node(node,ns,outfile):
  """
  Recursive function to extract information from IPC.
  
  :params:
    node (ElementTree.Element) : a node in XML
    ns (dict) : name space of element
    outfile (_io.TextIOWrapper) : file where to store parsed node
  """
  if node != None:
    
    label = node.attrib.get('symbol',None)
    
    if label:
    
      text = node.findall('./default:textBody/default:title/default:titlePart/default:text', namespaces = ns)
      desc = [t.text for t in text if t != None]
      children = [child.attrib.get('symbol') for child in node.getchildren() if child.tag == '{http://www.wipo.int/classifications/ipc/masterfiles}ipcEntry' ]
      to_dump = {'node' : label, 'children' : children, 'desc' : desc}
      outfile.write(json.dumps(to_dump)+'\n')
   
  for item in node.getchildren():
    rec_parse_ipc_node(item,ns,outfile)
  else:
    return 0
      
  
def parse_original_ipc_schme(ipc_path,outfile):
  """
  Helper function to parse IPC XML scheme. The file is parsed recursively and each node (label) is stored in a second file
  presenting its children and the node description for further processing.
  
  :params:
    ipc_path (str) : path to IPC in XML
    outfile (str) : path where to store parsed version
  """

  infile = ipc_path
  tree = etree.parse(infile)
  root = tree.getroot()
  
  logger.info("Start parsing original IPC XML file : `{}`".format(ipc_path))
  
  ns = {'default': 'http://www.wipo.int/classifications/ipc/masterfiles'}
  
  
  with open(outfile, 'w') as out:
    rec_parse_ipc_node(root,ns,out)
    
  logger.info("Completed parsing. Intermediate output has been saved at : `{}`".format(outfile))
    


if __name__ == "__main__":
  
  args = parse_arguments()
  
  logger.info("\nStart mapping creation process for IPC\n")
  
  nodes = parse_original_ipc_schme(args.ipc,args.parsed)
  
  flat_to_hier = get_mapping_flat_to_hier_from_parsed_ipc_scheme(args.parsed)
  
  flat_to_desc = get_mapping_flat_to_desc_from_parsed_ipc_scheme(args.parsed)
  
  mapping = merge_mappings(flat_to_hier,flat_to_desc)
  
  logger.info("Completed merging of (1) mappping : label -> description and (2) label -> hierarchical label.")
  logger.info("Result will be stored at `{}`".format(args.mapping))
  
  iom.save_pickle(args.mapping,mapping)
  
  
  
  
 