# DSC180A Project 1
In this project, we aim to introduce the novel transformer-based graph learning model and benchmark its performance against state-of-the-art graph neural networks (GNNs), specifically focusing on the CORA, IMDB-BINARY, and ENZYME datasets. We also seek to observe how varying parameters affects performance, such as length, width, depth, datasets, tasks, etc.

## Retrieving the Data
Data can be retrieved their through pytorch_geometric data library, or through `etl.load_data()`, which will import 3 datasets from the pytorch_geometric data library: CORA, ENZYME, and IMDB-BINARY.

## Running the Code
To install the dependencies, run the following command from the root directory of the project: pip install -r requirements.txt

#### Building and Benchmarking Models
Examples of how to implement and benchmark the models for the 3 datasets can be found in their respective .ipynb notebooks: GAT.ipynb, GCN.ipynb, and GIN.ipynb. Seeing how GAT and GCN compare across differing layer depths can be seen at GCN_vs_GAT_layers.ipynb.

##  File Descriptions
### eda.ipynb

### GAT.ipynb
Shows benchmark results for GAT architecture for 3 datasets.

### GCN.ipynb
Shows benchmark results for GCN architecture for 3 datasets.

### GIN.ipynb
Shows benchmark results for GIN architecture for 3 datasets.

### GCN_vs_GAT_layers.ipynb
Compares performance of GCN and GAT models on CORA dataset as layer numbers increases.

### etl.py
Contains source code:
* Class definitions for GNN models
    * 6 total - each with  __init__() and forward() methods
        * GCNGraph( hidden_channels, dataset, layers)
        * GCNNode(in_channels, hidden_channels, out_channels, num_layers)
        * GINGraph(in_channels, hidden_channels, out_channels, num_layers)
        * GINNode(in_channels, hidden_channels, out_channels, num_layers)
        * GATGraph(in_channels, hidden_channels, out_channels, heads, layers)
        * GINNode(in_channels, hidden_channels, out_channels, heads, num_layers)
* Training and Testing methods for graph and node classifier models
    * node_train(model, data, optimizer) -> torch_geometric.nn.model (Node)
    * node_test(model, data) -> test_acc, train_acc, pred
    * graph_train(model, train_loader, optimizer, criterion = F.nll_loss) -> torch_geometric.nn.model (Graph)
    * graph_test(model, loader, train_loader) ->  test_acc, train_acc
* Helper Functions
    * Visualization
        * visualize_graph(data, title) -> networkx graph
            * nodes are colored by degree
        * visualize_by_pred_class(pred, data, title') -> networkx graph
            * nodes are colored by prediction
    * Calculate statistics
        * calculate_network_statistics(G: graph) -> dict of degree, diameter, etc
        * get_data_description(dataset) -> None
            * prints basic data descriptions and stats for dataset
    * Preprocess Data (split train/test loader)
        * preprocess_data(dataset, onehot = False, batch_size = 64)
            * will one hot encode data if onehot = True
    * Loading 3 datasets (CORA, IMDB_BINARY, ENYZME)
        * load_data() -> imdb_dataset, cora_dataset, enzyme_dataset
    * Running performance tests for GAT and GCN architectures based on layers
        * mod_layers_GCN(dataset, layers, epochs=200) -> final test accuracy
        * mod_layers_GAT(dataset, layers, epochs=200) -> final test accuracy

