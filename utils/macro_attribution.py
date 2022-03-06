from captum.attr import Saliency, IntegratedGradients, InputXGradient
import copy
import dgl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import networkx as nx
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDepictor, rdMolDescriptors
rdDepictor.SetPreferCoordGen(True)
import rdkit

import torch
from utils import load_networkx

class Attribution:
    def __init__(self, model_name, attribution_type):
        """
        Initializes Attribution class.

        Args:
        model_name : str, name of the PyTorch model architecture.
        attribution_type : str, name of the attribution type.

        Attributes:
        self.model_name :  str, name of the PyTorch model architecture.
        self.att_type : str, name of the attribution type.
        self.model_instance : PyTorch model instance, pre-trained supervised model.
        self.device : str, device to be used for attribution.
        self.attr_weights : np array, attribution weights for macromoelcule graph.
        self.attributes : np array, fp attribution weights for macromoelcule graph.
        self.node_weights : np array, node weights for macromoelcule graph.
        self.dgl_graph : DGL graph, macromolecule graph 
        self.NX_GRAPHS : dict, dictionary of networkx graphs
        """
        self.model_name = model_name
        self.att_type = attribution_type

    def _model_forward(self, node_feats, input_graph):
        """
        Obtains forward model for attribution calculation.

        Args:
        node_feats : PyTorch tensor, features of all nodes.
        input_graph : DGL graph, macromolecule graph to calculate attribution.
        """
        bg = input_graph.to(self.device)
        
        bg.requires_grad = True
        node_feats.requires_grad = True
        
        if self.model_name in ['MPNN', 'AttentiveFP', 'Weave']:
            edge_feats = bg.edata.pop('e').to(self.device)
            edge_feats.requires_grad = True
            return self.model_instance(bg, node_feats, edge_feats)
        else:
            bg.edata.pop('e').to('cuda')
            return self.model_instance(bg, node_feats)

    def calculate_attribution(self, model_instance, input_graph):
        """
        Calculates attribution for input graph.

        Args:
        model_instance : PyTorch model instance, pre-trained supervised model.
        input_graph : DGL graph, macromolecule graph to calculate attribution.

        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_instance = model_instance
        
        tmp_graph = copy.deepcopy(input_graph).to(self.device)
        node_feats = tmp_graph.ndata.pop('h').to(self.device)
        
        self.model_instance.to(self.device)
        if self.att_type == 'integrated_gradients':
            att_function = IntegratedGradients(self._model_forward)
            self.attr_weights = att_function.attribute(
                node_feats, 
                additional_forward_args=tmp_graph, 
                internal_batch_size=len(node_feats),
                n_steps = 100)
        elif self.att_type == 'input_gradient':
            att_function = InputXGradient(self._model_forward)
            self.attr_weights = att_function.attribute(
                node_feats, 
                additional_forward_args=tmp_graph,)
        elif self.att_type == 'attention':
            edge_feats = tmp_graph.edata.pop('e').to('cuda')
            _, self.att_weights = self.model_instance(
                tmp_graph, node_feats, edge_feats, get_node_weight=True)
            return self.att_weights[0].to('cpu').detach().numpy()
        
        self.attributes = np.multiply(
            node_feats.to('cpu').numpy(), self.attr_weights.to('cpu').numpy())
        self.attributes[self.attributes < 0] = 0
        
        self.node_weights = self.normalize_node_weights(self.attributes.sum(axis=1))
    
    def normalize_node_weights(self, node_weights):
        return node_weights/max(node_weights)
    
    def map_graph_id(self):
        """Maps DGL graph to graph ID in networkx graph dictionary."""
        for graph_id in self.NX_GRAPHS:
            graph = self.NX_GRAPHS[graph_id]
            flag = 0
            try:
                for node_idx in range(len(graph.nodes)):
                    if np.sum(
                        np.array(graph.nodes[node_idx]['h']) == 
                        np.array(self.dgl_graph.nodes[node_idx].data['h'][0])) != 128:
                        flag = -1
                        break
                if flag == -1:
                    continue
                else:
                    self.graph_id = graph_id
                    break
            except:
                pass
    
    def get_nodes(self):
        """Obtains list of nodes from networkx graph."""
        self.map_graph_id()
        self.nodes_list = [
            self.NX_GRAPHS[self.graph_id].nodes[idx]['label'] 
            for idx in range(len(self.NX_GRAPHS[self.graph_id].nodes))]
    
    def get_colors(self):
        """Obtains colors for the graph, unique to each monomer type."""
        colors = ["#244486", "#A6A6A6", "#B12122"]
        cmap = LinearSegmentedColormap.from_list("mycmap", colors)

        color_palette=[cmap(i) for i in np.linspace(0, 1, len(set(self.nodes_list)))]
        return dict(zip(list(set(self.nodes_list)), color_palette))
        
    def plot_graph(self, input_graph, NX_GRAPHS):
        """
        Displays graph of macromolecule with node weights corresponding to node size.

        Args:
        input_graph : input DGL graph, macromolecule graph 
        NX_GRAPHS : dict, dictionary of networkx graphs
        """
        self.dgl_graph = input_graph
        self.NX_GRAPHS = NX_GRAPHS
        
        self.get_nodes()
        color_monomer = self.get_colors()
        
        print(dict(zip(range(len(self.nodes_list)), self.nodes_list)))
        print('Key Monomer is', self.nodes_list[np.argmax(self.node_weights)])
        
        fig, ax = plt.subplots()
        nx.draw_networkx(
            dgl.to_networkx(self.dgl_graph),
            arrows=False,
            node_size = 300*10**self.node_weights,
            node_color = [color_monomer[node] for node in self.nodes_list],
            font_size = 18,
            font_color = 'w',
            font_weight = 'bold',)

        plt.axis('off')
        ax.set_xlim([1.2*x for x in ax.get_xlim()])
        ax.set_ylim([1.2*y for y in ax.get_ylim()])
        plt.show()
        
    def plot_fp(self, node_idx):
        """
        Displays activated fingerprint.

        Args:
        node_idx : int, index of monomer
        """
        activated_fp = self.attributes[node_idx]
        activated_fp = activated_fp/max(activated_fp).reshape(1,-1)
        activated_fp = np.ma.masked_where(activated_fp == 0.00, activated_fp)
        
        print(node_idx, self.nodes_list[node_idx])
        print('Key Substructure in', self.nodes_list[node_idx], str(np.argmax(activated_fp)))
        
        colors = ["#244486", "#B2B2B2", "#B12122"]
        cmap = LinearSegmentedColormap.from_list("mycmap", colors)
        cmap.set_bad(color='white')

        fig, ax = plt.subplots()
        im = ax.imshow(activated_fp,
                      cmap=cmap, interpolation='none')
        ax.set_aspect(8.5)
        plt.xticks(fontsize=15)
        plt.yticks([],fontsize=15)
        ax.set_yticklabels([''])
        plt.xticks(fontsize=20)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('Weights', size = 18)

        plt.show()
        
    def display_substructure(self, monomer, MON_SMILES_PATH, RADIUS, N_BITS):
        """
        Generates objects for fingerprint substructure.

        Args:
        monomer : str, monomer name
        MON_SMILES_PATH : str, path of the monomer smiles table
        RADIUS : int, radius of ECFP
        N_BITS : int, number of bits of ECFP

        Returns:
        mol : RDKit mol object, mol object for the monomer
        bi : dict, dictionary of 'on' bits
        """
        df = pd.read_csv(MON_SMILES_PATH)
        smiles = df[df['Molecule'] == monomer]['SMILES'].tolist()[0]
        mol = Chem.MolFromSmiles(smiles)

        bi = {}
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=RADIUS, nBits = N_BITS, useChirality = True, bitInfo=bi)

        return mol, bi
