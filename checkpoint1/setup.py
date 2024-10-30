from gnn import GCNNode, GCNGraph
import gnn
import torch
import helper_funcs
import torch.nn.functional as F
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx # converts to networkx graph
import numpy as np
from torch_geometric.transforms import OneHotDegree
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import random

# set up seeds
seed = 123
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
