"""GNN models used for benchmarking"""
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
import os.path as osp
from typing import Any, Dict, Optional

from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import multiprocessing as mp

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
    def __init__(self, hidden_channels, num_node_features, num_classes, layers=2):
        super().__init__()
        self.layers = layers
        
        # Initialize the first convolutional layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        
        # Add additional convolutional layers
        for _ in range(1, layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Linear layer for the final classification
        self.lin = Linear(hidden_channels, num_classes)

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
    data.x = data.x.float()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return model

@torch.no_grad()
def node_test(model, data):
    '''returns test accuracy and train accuracy'''
    model.eval()
    data.x = data.x.float()
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
        data.x = data.x.float()
        output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return model

@torch.no_grad()
def graph_test(model, test_loader, train_loader=None):
    '''outputs test accuracy and optionally train accuracy'''
    model.eval()
    correct_test = 0

    # Calculate test accuracy
    for data in test_loader:
        data.x = data.x.float()
        output = model(data.x, data.edge_index, data.batch)
        pred = output.argmax(dim=1)
        correct_test += pred.eq(data.y).sum().item()
    test_acc = correct_test / len(test_loader.dataset)
    
    # Calculate train accuracy if train_loader is provided
    if train_loader is not None:
        correct_train = 0
        for data in train_loader:
            data.x = data.x.float()
            output = model(data.x, data.edge_index, data.batch)
            pred = output.argmax(dim=1)
            correct_train += pred.eq(data.y).sum().item()
        train_acc = correct_train / len(train_loader.dataset)
    else:
        train_acc = None  # If train_loader is not provided

    return test_acc, train_acc


### GRAPHGPS ###

# RedrawProjection class for PerformerAttention
class RedrawProjection:
    def __init__(self, model: torch.nn.Module, redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
        else:
            self.num_last_redraw += 1
            
# Define the GPS model
class GPSGraph(torch.nn.Module):
    def __init__(self, num_node_features:int, channels: int, num_layers: int, attn_type: str, attn_kwargs: Dict[str, Any]):
        super().__init__()
        self.node_emb = Linear(num_node_features or 5, channels)
        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            # Set edge_dim to 1 for dummy edge attributes
            conv = GPSConv(channels, GINEConv(nn, edge_dim=1), heads=4,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )
        self.redraw_projection = RedrawProjection(
            self.convs, redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)

        # Handle missing edge attributes
        if edge_attr is None:
            edge_attr = torch.ones((edge_index.size(1), 1), device=edge_index.device)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        x = global_add_pool(x, batch)
        return self.mlp(x)



def train_gps_graph(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        data.x = data.x.float()
        # Redraw projection matrices for Performer attention
        model.redraw_projection.redraw_projections()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out.squeeze(), data.y.float())
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test_gps_graph(model, loader, device):
    model.eval()
    total_correct = 0
    for data in loader:
        data = data.to(device)
        data.x = data.x.float()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds = (torch.sigmoid(out.squeeze()) > 0.5).long()
        total_correct += (preds == data.y).sum().item()
    return total_correct / len(loader.dataset)


class GPSNode(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, num_layers):
        super(GPSNode, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        out = self.lin(x)
        return out


def train_gps_nodes(model, data, optimizer, device):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    # Redraw projection matrices for Performer attention (if applicable)
    # model.redraw_projection.redraw_projections()
    data.x = data.x.float()
    out = model(data.x, data.edge_index)
    loss = torch.nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_gps_nodes(model, data, device):
    model.eval()
    data = data.to(device)
    data.x = data.x.float()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    # Calculate training accuracy
    train_correct = pred[data.train_mask] == data.y[data.train_mask]
    train_acc = int(train_correct.sum()) / int(data.train_mask.sum())

    # Calculate test accuracy
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())

    return train_acc, test_acc

class GPSLongRange(torch.nn.Module):
    """GPS for long range dataset"""
    def __init__(self, num_node_features, hidden_channels, num_classes, num_layers):
        super(GPSLongRange, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
    def forward(self, x, edge_index, batch):
        # Apply GCN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        # Global pooling 
        x = global_mean_pool(x, batch)
        # Final classification layer
        return self.lin(x)
# Train function
def longrange_train(model, data_loader, optimizer, device):
    """trains LongRange Classifier"""
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        data.x = data.x.float()  # check if x is float
        if data.y.ndim > 1 and data.y.size(1) > 1:
            data.y = data.y.argmax(dim=1) 
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch) 
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


# Test function
@torch.no_grad()
def longrange_test(model, data_loader, device):
    """returns testing accuracy for LongRange Classifier"""
    model.eval()
    correct = 0
    total = 0
    for data in data_loader:
        data = data.to(device)
        data.x = data.x.float()  # check if x is float
        if data.y.ndim > 1 and data.y.size(1) > 1:
            data.y = data.y.argmax(dim=1) 
        out = model(data.x, data.edge_index, data.batch) 
        pred = out.argmax(dim=1) 
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
    return correct / total  # Return accuracy