import matplotlib.pyplot as plt
import networkx as nx

def graph(tmp_graph):
    """Plots networkx graph."""
    node_list = [
        tmp_graph.nodes[idx]['label'] 
            for idx in range(len(tmp_graph.nodes))]
    nx.draw_networkx(tmp_graph)
    print(dict(zip(range(len(node_list)), node_list)))

def similarity_matrix(matrix):
    """Plots similarity matrix."""
    fig, ax = plt.subplots()

    plt.imshow(matrix,)
    plt.xlabel('Macromolecules', fontsize=18)
    plt.ylabel('Macromolecules', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.locator_params(axis='both', nbins=6)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Similarity', size=18)
    plt.show()

