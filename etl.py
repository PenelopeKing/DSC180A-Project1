"""
All GNN model classes types used in benchmarking
"""
from setup import *
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


### GNN CLASSES ###

# GAT code adjusted from pytorch-geometric exmaple data
class GATNode(torch.nn.Module):
    """for node classifcation"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # init layers list
        self.gat_layers = torch.nn.ModuleList()
        # input
        self.gat_layers.append(GATConv(in_channels, hidden_channels, heads=heads))
        # hidden
        for _ in range(1, num_layers - 1):
            self.gat_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        # output layer
        self.gat_layers.append(GATConv(hidden_channels * heads, out_channels, heads=1))  # Set heads=1 for the final layer

    def forward(self, x, edge_index):
        for layer in range(self.num_layers - 1):
            x = F.elu(self.gat_layers[layer](x, edge_index))
        # last layer has no activation function
        x = self.gat_layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)  # log_softmax for classification

# GAT code adjusted from pytorch-geometric exmaple data
class GATGraph(torch.nn.Module):
    """For graph classification"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads, layers=2):
        super().__init__()
        
        self.layers = layers
        
        # Initialize the list of GATConv layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads, dropout=0.6))
        
        # Add additional GATConv layers
        for _ in range(1, layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads, dropout=0.6))
        
        # The last convolution layer
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6))

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=0.6, training=self.training)
        
        # Pass through each GATConv layer
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
        
        # Global mean pooling
        x = global_mean_pool(x, batch)
        
        return F.log_softmax(x, dim=1)
    
'''class GATGraph(torch.nn.Module):
    """for graph classifcation"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index, batch):
        edge_index = edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch) # global mean pool for pooling
        return F.log_softmax(x, dim=1)'''


# code adjusted from example code
class GCNNode(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(GCNNode, self).__init__()
        self.num_layers = num_layers
        # init layers list
        self.gcn_layers = torch.nn.ModuleList()
        # input
        self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
        # hidden layer
        for _ in range(1, num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
        # output
        self.gcn_layers.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        # activation
        for i in range(self.num_layers - 1):
            x = F.relu(self.gcn_layers[i](x, edge_index))
        # last layer has no activation
        x = self.gcn_layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)  # log_softmax for classification tasks

class GCNGraph(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, layers=2):
        super().__init__()
        self.layers = layers
        
        # Initialize the first convolutional layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(dataset.num_node_features, hidden_channels))
        
        # Add additional convolutional layers
        for _ in range(1, layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Linear layer for the final classification
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # Pass through each convolutional layer with ReLU activation
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()
        
        # Global pooling
        x = global_mean_pool(x, batch)  
        
        # Dropout and final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
    
'''class GCNGraph(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)  
        # Final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x'''

class GINNode(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels
        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        temp = self.mlp(x)
        return F.log_softmax(temp, dim=-1)


# Code adapted from pytorch_geometric code
class GINGraph(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels
        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        temp = self.mlp(x)
        return F.log_softmax(temp, dim=-1)

# code adapted from pytorch_geometric code
class GINGraph(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            mlp = Sequential(
            Linear(in_channels, hidden_channels),
            BatchNorm(hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))
            in_channels = hidden_channels

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
        x = global_add_pool(x, batch)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
    

### TRAINING AND TESTING FUNCTIONS ###

def node_train(model, data, optimizer):
    '''one training pass, returns model'''
    # train model
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return model

def node_test(model, data):
    '''returns test accuracy and train accuracy'''
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=-1)
    
    # Test accuracy
    test_mask = data.test_mask
    if test_mask.sum() == 0: 
        test_acc = 0.0
    else:
        test_acc = (pred[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()

    # Train accuracy
    train_mask = data.train_mask
    if train_mask.sum() == 0:
        train_acc = 0.0
    else:
        train_acc = (pred[train_mask] == data.y[train_mask]).sum().item() / train_mask.sum().item()

    return test_acc, train_acc, pred

def graph_train(model, train_loader, optimizer, criterion = F.nll_loss):
    '''one trianing pass, returns model'''
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return model


def graph_test(model, loader, train_loader=None):
    '''outputs test accuracy and optionally train accuracy'''
    model.eval()
    correct_test = 0

    # Calculate test accuracy
    for data in loader:
        output = model(data.x, data.edge_index, data.batch)
        pred = output.argmax(dim=1)
        correct_test += pred.eq(data.y).sum().item()
    test_acc = correct_test / len(loader.dataset)
    
    # Calculate train accuracy if train_loader is provided
    if train_loader is not None:
        correct_train = 0
        for data in train_loader:
            output = model(data.x, data.edge_index, data.batch)
            pred = output.argmax(dim=1)
            correct_train += pred.eq(data.y).sum().item()
        train_acc = correct_train / len(train_loader.dataset)
    else:
        train_acc = None  # If train_loader is not provided

    return test_acc, train_acc


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
