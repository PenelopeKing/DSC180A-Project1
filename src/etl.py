from models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch_geometric.transforms import OneHotDegree
from torch_geometric.nn import  GINConv, GCNConv, GATConv, global_add_pool, global_mean_pool, MLP
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential, LeakyReLU
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx # converts to networkx graph
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid


### HELPER FUNCTIONS ###

def load_data():
    '''returns 3 datasets used for benchmarking:
    imdb_dataset, cora_dataset, enzyme_dataset'''
    imdb_dataset = TUDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY')
    cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
    enzyme_dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    return imdb_dataset, cora_dataset, enzyme_dataset

# helper functions
def visualize_graph(data, title="Graph"):
    '''graphs graph data using networkx'''
    # turn torch_geometric to networkx graph
    G = to_networkx(data, to_undirected=True)
    # get degrees / max degrees of each node
    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    node_colors = [degrees[node] / max_degree for node in G.nodes()]
    pos = nx.spring_layout(G, seed=42, k = 0.45)  
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
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, test_loader

def visualize_by_pred_class(pred, data, title = 'Graph'):
    node_color = pred.cpu().numpy()

    def visualize_graph2(data, title=title):
        # turn torch_geometric to networkx graph
        G = to_networkx(data, to_undirected=True)
        # get degrees / max degrees of each node
        degrees = dict(G.degree())
        max_degree = max(degrees.values())
        pos = nx.spring_layout(G, seed=42, k = 0.45)  
        # plot graph
        plt.figure(figsize=(12, 12))
        nx.draw(G, pos, with_labels=False, node_color=node_color, cmap=plt.cm.rainbow, 
                node_size=40, edge_color="gray", alpha = 0.7)
        plt.title(title) # title
        plt.show()
        return G
    
    G = visualize_graph2(data, title=title)
    return G


def mod_layers_GCN(dataset, layers, epochs=200):
    hidden_channels = 32
    model = GCNNode(dataset.num_features, hidden_channels,
                dataset.num_classes, layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    for _ in range(epochs):
        _ = node_train(model, dataset, optimizer)
    test_acc, train_acc, pred = node_test(model, dataset)
    return test_acc

def mod_layers_GAT(dataset, layers, epochs=200):
    hidden_channels = 32
    model = GATNode(dataset.num_features, hidden_channels,
                dataset.num_classes, 4, layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    for _ in range(epochs):
        _ = node_train(model, dataset, optimizer)
    test_acc, train_acc, pred = node_test(model, dataset)
    return test_acc


def get_data_description(dataset):
    # Basic Information
    print(f'Dataset: {dataset}')  # Dataset name
    print(f'Number of graphs: {len(dataset)}')  # Number of graphs in the dataset
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Number of node features: {dataset.num_node_features}')
