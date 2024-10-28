import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx # converts to networkx graph
import numpy as np
from torch_geometric.transforms import OneHotDegree
from torch_geometric.loader import DataLoader

# helper functions
def visualize_graph(data, title="Graph"):
    # turn torch_geometric to networkx graph
    G = to_networkx(data, to_undirected=True)
    # get degrees / max degrees of each node
    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    node_colors = [degrees[node] / max_degree for node in G.nodes()]
    pos = nx.spring_layout(G, seed=42, k = 0.45)  # You can change the seed for different layouts
    # plot graph
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=False, node_color=node_colors, cmap=plt.cm.rainbow, 
            node_size=40, edge_color="gray", alpha = 0.7)
    plt.title(title) # title
    plt.show()
    return G

def calculate_network_statistics(G):
    degrees = dict(G.degree()) # get all degrees
    avg_degree = np.mean(list(degrees.values())) # average degree
    diameter = nx.diameter(G) # network diameter
    avg_path_length = nx.average_shortest_path_length(G) # average path length
    avg_clustering = nx.average_clustering(G) # average clustering coefficient

    return {
        'Average Degree': avg_degree,
        'Network Diameter': diameter,
        'Average Path Length': avg_path_length,
        'Average Clustering Coefficient': avg_clustering
    }

def preprocess_data(dataset, onehot = False, batch_size = 64):
    """one hot encodes data without node labels
    and splits into train test split 80/20"""
    if onehot: 
        transform = OneHotDegree(max_degree=300)
        dataset.transform = transform
    dataset = dataset.shuffle()
    n = len(dataset)
    split = int(n*0.8)
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    return train_loader, test_loader