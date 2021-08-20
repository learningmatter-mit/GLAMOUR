from __future__ import absolute_import

import os
import numpy as np
import networkx as nx
import re
import collections

import pandas as pd

from rdkit.Chem import rdMolDescriptors
from rdkit import Chem

import grakel
import dgl
import torch
from dgl.data.utils import save_graphs, load_graphs
from dgl.data import DGLDataset


def _txtread_index0(TXT_DATA_PATH, file):
    '''
    Processes .txt file for macromolecule into node and edge dictionaries
    
    Args:
    TXT_DATA_PATH: str, path to .txt files for dataset to be used for training
    file: str, name of .txt file in TXT_DATA_PATH directory to read
            
    Returns:
    [node_dict, edge_dict]: list, list of two dictionaries, one for nodes and the other for edges
    '''
    node_dict = {} 
    edge_dict = {} 
    with open(os.path.join(TXT_DATA_PATH,file)) as txt_file:
        monbool = False
        bondbool = False
        for line in txt_file: 
            line = line.strip() 
            if line == 'MONOMERS':
                monbool = True 
                continue 
            if line == '': 
                continue
            if line == 'BONDS':
                monbool = False
                bondbool = True 
                continue 
            if monbool == True:
                line_split = line.split(' ') 
                node_dict[int(line_split[0]) - 1] = line_split[1]
            if bondbool == True:
                line_split = line.split(' ') 
                pos_tuple = (int(line_split[0]) - 1, 
                             int(line_split[1]) - 1)
                edge_dict[pos_tuple] = line_split[2]
    return [node_dict, edge_dict]



def _graphgen(TXT_DATA_PATH, file):
    '''
    Processes .txt file for macromolecule into unfeaturized NetworkX graph
    
    Args:
    TXT_DATA_PATH: str, path to .txt files for dataset to be used for training
    file: str, name of .txt file in TXT_DATA_PATH directory to read
            
    Returns:
    graph: NetworkX graph, NetworkX graph corresponding with specified file in TXT_DATA_PATH
    '''
    dict_list = _txtread_index0(TXT_DATA_PATH, file)
    graph = nx.DiGraph()
    graph.add_nodes_from(list(dict_list[0].keys())) 
    graph.add_edges_from(list(dict_list[1].keys())) 
    for node in list(dict_list[0].keys()): 
        graph.nodes[node]['label'] = dict_list[0][node] 
    for edge in list(dict_list[1].keys()): 
        graph.edges[edge]['label'] = dict_list[1][edge] 
    return graph




def _mon_graphsgen(TXT_DATA_PATH):
    '''
    Processes all .txt files of a macromolecule type into a dictionary of unfeaturized NetworkX graphs
    
    Args:
    TXT_DATA_PATH: str, path to .txt files for dataset to be used for training
            
    Returns:
    mon_ordered: dict, sorted dictionary with keys as glycan IDs and values as NetworkX graphs
    '''
    mon_graphs = {}
    for subdir, dirs, files in os.walk(TXT_DATA_PATH):
        for file in files:
            if file in files and file.endswith('.txt'):
                glycan_id = file.split('_')[0]
                mon_graphs[glycan_id] = _graphgen(os.path.join(subdir),file)
    mon_ordered = collections.OrderedDict(sorted(mon_graphs.items()))
    return mon_ordered

def _df_to_smiles_dict(df, radius, nBits, useChirality=True):
    '''
    Generates dict of all monomers/bonds that comprise macromolecule and corresponding molecular fingerprints
    
    Args:
    df: DataFrame, DataFrame of all monomers/bonds that comprise macromolecule and corresponding SMILES         
    radius: int, radius of topological exploration 
    nBits: int, size of fingerprint bit-vector
    useChirality: boolean, whether to consider chirality in molecular fingerprint
    
    Returns:
    fp_dict: dict, dictionary of molecular fingerprint for each molecule in dataframe df
    '''
    fp_dict = {}
    for i in range(df.shape[0]):
        fp_dict[df['Molecule'].iloc[i]] = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(df['SMILES'].iloc[i]), 
                radius = radius,
                nBits = nBits,
                useChirality = useChirality
            )
    return fp_dict


def _monbond_to_onehot(df):
    '''
    Generates dict of all monomers/bonds that comprise macromolecule and corresponding one-hot encodings
    
    Args:
    df: DataFrame, DataFrame of all monomers/bonds that comprise macromolecule and corresponding SMILES
        
    Returns:
    onehot_dict: dict, dictionary of one-hot encoding for each molecule in dataframe df
    '''
    mol_list = df['Molecule'].tolist()
    onehot_dict = {}
    for mol in range(len(mol_list)):
        onehot = [0 for a in range(len(mol_list))]
        onehot[mol] = 1
        onehot_dict[mol_list[mol]] = onehot
    return onehot_dict


def _FpFeaturizer(dict_graphs, MON_SMILES, BOND_SMILES, FP_RADIUS_MON = 3, FP_RADIUS_BOND = 3, FP_BITS_MON = 128, FP_BITS_BOND = 16):
    '''
    Adds molecular fingerprint node/edge attributes to NetworkX graphs in dictionary
    Removes any graphs that produce errors during featurization
    
    Args:
    dict_graphs: dict, dictionary of all NetworkX graphs generated using _mon_graphsgen() 
    MON_SMILES: str, path to .txt file of all monomers that comprise macromolecule and corresponding SMILES
    BOND_SMILES: str, path to .txt file of all bonds that comprise macromolecule and corresponding SMILES
    FP_RADIUS_MON: int, radius of topological exploration for monomer fingerprint (default = 3)
    FP_RADIUS_BOND: int, radius of topological exploration for bond fingerprint (default = 3)
    FP_BITS_MON: int, size of fingerprint bit-vector for monomer (default = 128)
    FP_BITS_BOND: int, size of fingerprint bit-vector for bond (default = 16)
        
    Returns:
    featurized_dict: dict, dictionary of molecular fingerprint-featurized graphs for all glycan IDs
    '''
    df_monomer_smiles = pd.read_csv(MON_SMILES)
    df_bonds_smiles = pd.read_csv(BOND_SMILES)
    dict_fp_nodes = _df_to_smiles_dict(df_monomer_smiles, FP_RADIUS_MON, FP_BITS_MON)
    dict_fp_edges = _df_to_smiles_dict(df_bonds_smiles, FP_RADIUS_BOND, FP_BITS_BOND)
    dict_fp = {}
    dict_fp.update({key: np.array(dict_fp_nodes[key]) for key in dict_fp_nodes})
    dict_fp.update({key: np.array(dict_fp_edges[key]) for key in dict_fp_edges})
    
    featurized_dict = {}
    
    for idnum in dict_graphs:
        graph = dict_graphs[idnum]
        add_dict = True
        for node in graph.nodes:
            try:
                graph.nodes[node]['h'] = torch.FloatTensor(dict_fp[graph.nodes[node]['label']])
            except:
                add_dict = False
        for edge in graph.edges:
            try:
                graph.edges[edge]['e'] = torch.FloatTensor(dict_fp[graph.edges[edge]['label']])
            except:
                add_dict = False
        if add_dict == True:
            featurized_dict[idnum] = graph
    
    return featurized_dict


def _OnehotFeaturizer(dict_graphs, MON_SMILES, BOND_SMILES):
    '''
    Adds one-hot encoding node/edge attributes to NetworkX graphs in dictionary
    Removes any graphs that produce errors during featurization
    
    Args:
    dict_graphs: dict, dictionary of all NetworkX graphs generated using _mon_graphsgen() 
    MON_SMILES: str, path to Dataframe of macromolecule monomers and corresponding SMILESstr, path to .txt file of all monomers that comprise macromolecule and corresponding SMILES
    BOND_SMILES: str, path to .txt file of all bonds that comprise macromolecule and corresponding SMILES
        
    Returns:
    featurized_dict: dict, dictionary of one-hot encoding-featurized graphs for all molecule IDs
    '''
    df_monomer_smiles = pd.read_csv(MON_SMILES)
    df_bonds_smiles = pd.read_csv(BOND_SMILES)
    dict_onehot_nodes = _monbond_to_onehot(df_monomer_smiles)
    dict_onehot_edges = _monbond_to_onehot(df_bonds_smiles)
    dict_onehot = {}
    dict_onehot.update({key: np.array(dict_onehot_nodes[key]) for key in dict_onehot_nodes})
    dict_onehot.update({key: np.array(dict_onehot_edges[key]) for key in dict_onehot_edges})
    
    featurized_dict = {}
    
    for idnum in dict_graphs:
        graph = dict_graphs[idnum]
        add_dict = True
        for node in graph.nodes:
            try:
                graph.nodes[node]['h'] = torch.FloatTensor(dict_onehot[graph.nodes[node]['label']])
            except:
                add_dict = False
        for edge in graph.edges:
            try:
                graph.edges[edge]['e'] = torch.FloatTensor(dict_onehot[graph.edges[edge]['label']])
            except:
                add_dict = False
        if add_dict == True:
            featurized_dict[idnum] = graph
            
    return featurized_dict


def networkx_feat(TXT_DATA_PATH, MON_SMILES, BOND_SMILES, FEAT = 'fp', FP_RADIUS_MON = None, FP_RADIUS_BOND = None, FP_BITS_MON = None, FP_BITS_BOND = None):
    '''
    Processes all .txt files of a macromolecule type into a dictionary of featurized NetworkX graphs
    
    Args:
    TXT_DATA_PATH: str, path to .txt files for dataset to be used for training
    MON_SMILES: str, path to .txt file of all monomers that comprise macromolecule and corresponding SMILES
    BOND_SMILES: str, path to .txt file of all bonds that comprise macromolecule and corresponding SMILES
    FEAT: str, type of attribute with which to featurizer nodes and edges of macromolecule NetworkX graphs (default = 'fp')
        
    Returns:
    graphs_feat: dict, dictionary of molecular fingerprint-featurized graphs for all molecule IDs
    '''
    mon_graphs = {}
    mon_graphs = _mon_graphsgen(TXT_DATA_PATH)
    if FEAT == 'fp':
        graphs_feat = _FpFeaturizer(mon_graphs, MON_SMILES, BOND_SMILES, FP_RADIUS_MON, FP_RADIUS_BOND, FP_BITS_MON, FP_BITS_BOND)
    elif FEAT == 'onehot':
        graphs_feat = _OnehotFeaturizer(mon_graphs, MON_SMILES, BOND_SMILES)
    return graphs_feat

