import os
import grakel
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn import decomposition, manifold
import umap.umap_ as umap

import warnings
warnings.filterwarnings("ignore")

def edit_distance(graph1, graph2, node_attr='h', edge_attr='e', upper_bound=100, indel_mul=3, sub_mul=3):
    """
    Calculates exact graph edit distance between 2 graphs.

    Args:
    graph1 : networkx graph, graph with node and edge attributes 
    graph2 : networkx graph, graph with node and edge attributes 
    node_attr : str, key for node attribute
    edge_attr : str, key for edge attribute
    upper_bound : int, maximum edit distance to consider
    indel_mul: float, insertion/deletion cost
    sub_mul: float, substitution cost

    Returns:
    np.float, distance, how similar graph1 is to graph2
    """
    def node_substitution_scoring(dict_1, dict_2):
        """Calculates node substitution score."""
        multiplier = sub_mul if distance.rogerstanimoto(
            dict_1[node_attr], dict_2[node_attr]) != 0 else 0
        return multiplier*(1 - distance.rogerstanimoto(
            dict_1[node_attr], dict_2[node_attr]))

    def edge_substitution_scoring(dict_1, dict_2):
        """Calculates edge substitution score."""
        multiplier = sub_mul if distance.rogerstanimoto(
            dict_1[edge_attr], dict_2[edge_attr]) != 0 else 0
        return multiplier*(1 - distance.rogerstanimoto(
            dict_1[edge_attr], dict_2[edge_attr]))
    
    def constant_value(dict_1):
        """Returns constant score for insertion/deletion."""
        return indel_mul

    graph1 = feature_conversion(graph1, node_attr, edge_attr)
    graph2 = feature_conversion(graph2, node_attr, edge_attr)

    return min(
        nx.optimize_graph_edit_distance(
        graph1, graph2, 
            node_subst_cost = node_substitution_scoring,
            edge_subst_cost = edge_substitution_scoring,
            upper_bound  = upper_bound,
            node_del_cost = constant_value, 
            node_ins_cost = constant_value, 
            edge_del_cost = constant_value, 
            edge_ins_cost = constant_value, 
        ))

def feature_conversion(graph, node_attr, edge_attr):
    """Converts networkx graph features from tensors to np array."""
    for node in graph.nodes:
        graph.nodes[node][node_attr] = np.array(graph.nodes[node][node_attr])
    for edge in graph.edges:
        graph.edges[edge][edge_attr] = np.array(graph.edges[edge][edge_attr])
    return graph

def similarity_matrix(dict_graphs, method='kernel', **kwargs):
    """
    Calculates an (n x n) similarity matrix for a dictionary of macromolecule networkx graphs.
    
    Args:
    dict_graphs : dict, dictionary of networkx graphs, key: graph_id, value: networkx graph object
    method : str, similarity matrix calculation method - 'kernel' or 'exact_distance'
    **kwargs : optional arguments for similarity matrix method

    Returns:
    matrix : np array, n x n similarity matrix
    """

    node_attr = 'h' if 'node_attr' not in kwargs else kwargs['node_attr']
    edge_attr = 'e' if 'edge_attr' not in kwargs else kwargs['edge_attr']
    upper_bound = 100 if 'upper_bound' not in kwargs else kwargs['upper_bound']

    list_graphs = [feature_conversion(
        graph, node_attr, edge_attr) for graph in list(dict_graphs.values())]

    if method == 'kernel':
        grakel_graphs = grakel.graph_from_networkx(
            list_graphs, node_labels_tag=node_attr, edge_labels_tag=edge_attr)
        gk = grakel.PropagationAttr(
            n_jobs=-1 if 'n_jobs' not in kwargs else kwargs['n_jobs'], 
            t_max=upper_bound,  
            random_state=108 if 'random_state' not in kwargs else kwargs['random_state'],)
        return gk.fit_transform(grakel_graphs)

    elif method == 'exact_distance':
        matrix = np.zeros((len(list_graphs), len(list_graphs)))
        for idx_ref, ref_graph in enumerate(list_graphs):
            for idx_seq, tmp_graph in enumerate(list_graphs):
                if idx_ref >= idx_seq:
                    distance = edit_distance(
                        ref_graph, tmp_graph, node_attr, edge_attr, upper_bound)
                    matrix[idx_ref, idx_seq] = distance
                    matrix[idx_seq, idx_ref] = distance

        return matrix

def edit_distance(graph1, graph2, node_attr, edge_attr, upper_bound):
    """
    Calculates exact graph edit distance between 2 graphs.

    Args:
    graph1 : networkx graph, graph with node and edge attributes 
    graph2 : networkx graph, graph with node and edge attributes 
    node_attr : str, key for node attribute
    edge_attr : str, key for edge attribute
    upper_bound : int, maximum edit distance to consider

    Returns:
    np.float, distance, how similar graph1 is to graph2
    """
    def node_substitution_scoring(dict_1, dict_2):
        return 1 - distance.rogerstanimoto(
            dict_1[node_attr], dict_2[node_attr])

    def edge_substitution_scoring(dict_1, dict_2):
        return 1 - distance.rogerstanimoto(
            dict_1[edge_attr], dict_2[edge_attr])
    
    def constant_value(dict_1):
        return 5

    graph1 = feature_conversion(graph1, node_attr, edge_attr)
    graph2 = feature_conversion(graph2, node_attr, edge_attr)

    return min(
        nx.optimize_graph_edit_distance(
        graph1, graph2, 
            node_subst_cost = node_substitution_scoring,
            edge_subst_cost = edge_substitution_scoring,
            upper_bound  = upper_bound,
            node_del_cost = constant_value, 
            node_ins_cost = constant_value, 
            edge_del_cost = constant_value, 
            edge_ins_cost = constant_value, 
        ))

def dimensionality_reduction(matrix, method, **kwargs):
    """
    Reduces dimensionality of similarity matrix.

    Args:
    matrix : np array, similarity matrix
    method : str, method of dimensionality reduction
    **kwargs : optional arguments for dimensionality reduction method

    Returns:
    embedding : np array, (n_samples, n_components) after the dimensionality reduction
    """
    if method == 'umap':
        reducer = umap.UMAP(
            n_components=2 if 'n_components' not in kwargs else kwargs['n_components'], 
            n_neighbors=4 if 'n_neighbors' not in kwargs else kwargs['n_neighbors'], 
            random_state=108 if 'random_state' not in kwargs else kwargs['random_state'], 
            metric = 'precomputed' if 'metric' not in kwargs else kwargs['metric'])

    elif method == 'tsne':
        reducer = manifold.TSNE(
            n_components=2 if 'n_components' not in kwargs else kwargs['n_components'], 
            perplexity=10 if 'perplexity' not in kwargs else kwargs['perplexity'], 
            n_iter=1000  if 'n_iter' not in kwargs else kwargs['n_iter'],
            n_jobs=-1 if 'n_jobs' not in kwargs else kwargs['n_jobs'], 
            random_state=108 if 'random_state' not in kwargs else kwargs['random_state'],)
    return reducer.fit_transform(matrix)

def plot_embeddings(embeddings, NX_GRAPHS, DF_PATH, method):
    """
    Plots 2D component embeddings obtained from dimensionality reduction.

    Args:
    embeddings : np array, (n_samples, n_components) after the dimensionality reduction
    NX_GRAPHS :  dict, dictionary of networkx graphs, key: graph_id, value: networkx graph object
    DF_PATH : str, path of dataframe with labels
    method : str, method for xlabel, ylabel
    """
    df = pd.read_csv(DF_PATH)
    immunogenic_idx = []
    non_immunogenic_idx = []
    colors = []

    for idx, graph_id in enumerate(NX_GRAPHS):
        if df[df.ID == graph_id]['Immunogenic'].tolist()[0] == 'Yes':
            immunogenic_idx.append(idx)
            colors.append('#B12122')
        else:
            non_immunogenic_idx.append(idx)
            colors.append('#2C7FFF')
    colors = np.array(colors)

    fig = plt.figure()
    plt.scatter(
        embeddings[:, 0][immunogenic_idx],
        embeddings[:, 1][immunogenic_idx],
        s = 100,
        c = colors[immunogenic_idx],
        alpha = 0.5,
        label = 'Immunogenic')
    plt.scatter(
        embeddings[:, 0][non_immunogenic_idx],
        embeddings[:, 1][non_immunogenic_idx],
        s = 100,
        c = colors[non_immunogenic_idx],
        alpha = 0.5,
        label = 'Non-Immunogenic')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(markerscale = 1, fontsize=12, 
               borderpad=0.2, labelspacing=0.2, handletextpad=0.2, handlelength=1)
    plt.locator_params(nbins=5)
    plt.xlabel(method + ' C$\mathregular{_1}$', fontsize=20)
    plt.ylabel(method + ' C$\mathregular{_2}$', fontsize=20)
    plt.show()
