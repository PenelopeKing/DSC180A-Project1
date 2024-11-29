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
import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.datasets import LRGBDataset
import sys
import os
import random
from torch.utils.data import DataLoader, TensorDataset
seed = 123
from torch_geometric.loader import NeighborLoader
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.data import DataLoader, Dataset
import torch_geometric

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import multiprocessing as mp


### HELPER FUNCTIONS ###

class ListToDataset(Dataset):
    """Wraps a list of Data objects into a Dataset object."""
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
### END OF HELPER FUNCTIONS ###

# Load Peptides-func dataset
def load_peptides_func(parallel=True, subset_ratio=1.0):
    """Preprocess and train-valid-test split Peptides func LRGB data."""
    dataset = LRGBDataset(root='/tmp/Peptides-func', name='Peptides-func')

    # Train-test split (80-20 split)
    train_split = int(len(dataset) * 0.8)
    train_dataset = dataset[:train_split]
    test_dataset = dataset[train_split:]

    # Reduce dataset size for faster testing if subset ratio is less than 100%
    if subset_ratio < 1.0:
        import random
        random.seed(42)
        train_dataset = random.sample(train_dataset, max(1, int(len(train_dataset) * subset_ratio)))
        test_dataset = random.sample(test_dataset, max(1, int(len(test_dataset) * subset_ratio)))

    print('Loaded and split data')
    
    # Configure DataLoader
    num_workers = max(1, mp.cpu_count() - 2) if parallel else 0
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True)
    num_classes = 10 
    return train_loader, test_loader, num_classes
    
def load_imdb():
    """Preprocess and train-valid-test split IMDB data."""
    class AddRandomNodeFeatures(T.BaseTransform):
        def __call__(self, data):
            if data.x is None:
                data.x = torch.randn((data.num_nodes, 5))  # Random 5D features
            return data
    
    transform = T.Compose([
    T.OneHotDegree(max_degree=135),  # Add degree-based features
    AddRandomNodeFeatures(),         # Add random features if x is missing
    ])

    imdb_dataset = TUDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY')
    imdb_dataset.transform = transform
    perm = torch.randperm(len(imdb_dataset))
    train_idx = perm[:int(0.8 * len(imdb_dataset))]
    test_idx = perm[int(0.8 * len(imdb_dataset)):]
    train_dataset = imdb_dataset[train_idx]
    test_dataset = imdb_dataset[test_idx]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    return train_loader, test_loader
    
    
def load_enzyme():
    """preprocess and train-valid-test split enzyme data"""
    enzyme_dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    perm = torch.randperm(len(enzyme_dataset))
    train_idx = perm[:int(0.8 * len(enzyme_dataset))]
    test_idx = perm[int(0.8 * len(enzyme_dataset)):]
    train_dataset = enzyme_dataset[train_idx]
    test_dataset = enzyme_dataset[test_idx]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    return train_loader, test_loader

def load_cora():
    """Preprocess and train-valid-test mask split CORA data"""
    cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')[0]  # CORA dataset object
    num_nodes = cora_dataset.num_nodes
    perm = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    train_indices = perm[:train_size]
    test_indices = perm[train_size:]

    # Initialize train and test masks
    cora_dataset.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    cora_dataset.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Set the train and test masks
    cora_dataset.train_mask[train_indices] = True
    cora_dataset.test_mask[test_indices] = True

    # Create train and test datasets
    train_data = TensorDataset(
        torch.arange(num_nodes)[cora_dataset.train_mask],  # Node indices for training
        cora_dataset.x[cora_dataset.train_mask],  # Features
        cora_dataset.y[cora_dataset.train_mask],  # Labels
    )

    test_data = TensorDataset(
        torch.arange(num_nodes)[cora_dataset.test_mask],  # Node indices for testing
        cora_dataset.x[cora_dataset.test_mask],  # Features
        cora_dataset.y[cora_dataset.test_mask],  # Labels
    )

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    return cora_dataset, train_loader, test_loader

def load_data():
    '''returns 3 datasets used for benchmarking:
    imdb_dataset, cora_dataset, enzyme_dataset'''
    imdb_dataset = TUDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY')
    cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
    enzyme_dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    return imdb_dataset, cora_dataset, enzyme_dataset

def load_data_all():
    '''returns 3 datasets used for benchmarking:
    imdb_dataset, cora_dataset, enzyme_dataset, cocosp dataset'''
    imdb_dataset = TUDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY')
    cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
    enzyme_dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    peptide_dataset = LRGBDataset(root='/tmp/Peptides-func', name='Peptides-func')
    return imdb_dataset, cora_dataset, enzyme_dataset, peptide_dataset

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

    n = len(dataset)
    train_split = int(n * 0.8)
    #val_split = int(n * 0.9)

    # Split the dataset
    train_dataset = dataset[:train_split]
    #val_dataset = dataset[train_split:val_split]
    test_dataset = dataset[train_split:]

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
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
