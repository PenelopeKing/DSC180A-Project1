import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch_geometric.transforms import OneHotDegree
from torch_geometric.nn import  GINConv, GCNConv, GATConv, global_add_pool, global_mean_pool, MLP
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential, LeakyReLU


torch.manual_seed(12345)

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


# adjusted from pytorch-gemeotric examples
def GATNode_train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

def GATNode_test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


# GAT code adjusted from pytorch-geometric exmaple data
class GATGraph(torch.nn.Module):
    """for graph classifcation"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, data):
        x = data.x 
        edge_index = data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, data.batch) # global mean pool for pooling
        return F.log_softmax(x, dim=1)


def GATGraph_train(model, train_loader, train_dataset, optimizer):
    model.train()
    total_loss = 0.
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)

def GATGraph_test(model, loader):
    model.eval()
    total_correct = 0
    for data in loader:
        out = model(data)
        pred = out.max(dim=1)[1]
        total_correct += pred.eq(data.y).sum().item()
    return total_correct / len(loader.dataset)



# code adjusted from example code
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()
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

    def forward(self, x, edge_index, data):
        # activation
        for i in range(self.num_layers - 1):
            x = F.relu(self.gcn_layers[i](x, edge_index))
        # last layer has no activation
        x = self.gcn_layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)  # log_softmax for classification tasks


def train_gcn(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


def test_gcn(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data).argmax(dim=-1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs



# code from pytorch-geometric documentation
class GCNNode(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
def GCNNode_train(model, data, optimizer):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return model

class GCNGraph(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

def GCNGraph_train(model, criterion, optimizer, train_loader):
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def GCNGraph_test(model, loader):
     model.eval()
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  
     return correct / len(loader.dataset)  



# code adapted from pytorch_geometric code
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
    

def GIN_train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def GIN_test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

# code adapted from pytorch_geometric code
class GIN2(torch.nn.Module):
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