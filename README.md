# DSC180A Project 1
In this project, I aim to introduce the a transformer-based graph learning model and benchmark its performance against state-of-the-art graph neural networks (GNNs), specifically focusing on the CORA, IMDB-BINARY, ENZYME, and PEPTIDE-FUNC datasets. We also seek to observe how varying parameters affects performance, such as length, width, depth, datasets, tasks, etc. The models we are benchmarking are the following: GCN, GIN, GAT, and GraphGPS. Each have their own unique pros and cons and differing mechanisms.

## Retrieving the Data
Data can be retrieved their through pytorch_geometric data library, or through `etl.load_data_all()`, which will import 4 datasets from the pytorch_geometric data library: CORA, ENZYME, IMDB-BINARY, and PEPTIDE-FUNC.

## Running the Code
To install the dependencies, run the following command from the root directory of the project: `pip install -r requirements.txt`

Alternatively, you can also create a conda environment by running: `conda env create -f environment.yml`

* The `src` directory contains all my code used to build and train the models

#### Building and Benchmarking Models
Examples of how to implement and benchmark the models for the 4 datasets can be found in their respective .ipynb notebooks: GAT.ipynb, GCN.ipynb, GPS.ipynb,  GIN.ipynb, and long_range.ipynb. Seeing how GAT and GCN compare across differing layer depths can be seen at GCN_vs_GAT_layers.ipynb.

##  File Descriptions
In `/notebooks/` you will find the best use cases for the code and models. The 3 main datasets (ENZYME, CORA, IMDB-BINARY) are all analyzed in the same notebooks in their respective model's notebook. The longe range graph benchmark dataset (from https://arxiv.org/abs/2206.08164) is analyzed in its own special notebook `long_range.ipynb`.

### eda.ipynb
Basic EDA of the datasets.

### GAT.ipynb
Shows benchmark results for GAT architecture for 3 benchmark datasets (no long range).

### GCN.ipynb
Shows benchmark results for GCN architecture for 3 benchmark datasets (no long range).

### GIN.ipynb
Shows benchmark results for GIN architecture for 3 benchmark datasets (no long range).

### GPS.ipynb
Shows benchmark results for GraphGPS architecture for 3 benchmark datasets (no long range).

### GCN_vs_GAT_layers.ipynb
Compares performance of GCN and GAT models on CORA dataset as layer numbers increases.

### long_range.ipynb
Shows benchmark results for GAT, GCN, GraphGPS, and GIN for the PEPTIDE-FUNC long range benchmark dataset.

### models.py
Contains models used for benchmarking
* Class definitions for GNN models
    * All models have own  __init__() and forward() methods (NOTE: in_channels = features, out_channels = classes and vice versa)
        * GCNGraph( hidden_channels, dataset, layers)
        * GCNNode(in_channels, hidden_channels, out_channels, num_layers)
        * GINGraph(in_channels, hidden_channels, out_channels, num_layers)
        * GINNode(in_channels, hidden_channels, out_channels, num_layers)
        * GATGraph(in_channels, hidden_channels, out_channels, heads, layers)
        * GINNode(in_channels, hidden_channels, out_channels, heads, num_layers)
        * GPSGraph(num_node_features, channels, num_layers, attn_type, attn_kwargs)
        * GPSNode(num_node_features, hidden_channels, num_classes, num_layers))
* Training and Testing methods for graph and node classifier models -- note that GraphGPS and the PEPTIDE-FUNC long range dataset have their own special train/test functions.
    * node_train(model, data, optimizer) -> torch_geometric.nn.model (Node)
    * node_test(model, data) -> test_acc, train_acc, pred
    * graph_train(model, train_loader, optimizer, criterion = F.nll_loss) -> torch_geometric.nn.model (Graph)
    * graph_test(model, loader, train_loader) ->  test_acc, train_acc
    * train_gps_nodes(model, data, optimizer, device): for GPSNode ONLY
    * test_gps_nodes(model, data, device)L for GPSNode oNLY
    * train_gps_graph(model, train_loader, optimizer, device): for GPSGraph ONLY
    * test_gps_graph(model, loader, device): for GPSGraph ONLY
    * longrange_train(model, data_loader, optimizer, device): for any model training on long range dataset
    * longrange_test(model, data_loader, device): for any model training on long range dataset
### etl.py
* Helper Functions:
    * Visualization:
        * visualize_graph(data, title) -> networkx graph
            * nodes are colored by degree
        * visualize_by_pred_class(pred, data, title') -> networkx graph
            * nodes are colored by prediction
    * Calculate statistics:
        * calculate_network_statistics(G: graph) -> dict of degree, diameter, etc
        * get_data_description(dataset) -> None
            * prints basic data descriptions and stats for dataset
    * Preprocess Data (split train/test loader):
        * preprocess_data(dataset, onehot = False, batch_size = 64)
            * will one hot encode data if onehot = True
    * Loading 3 datasets (CORA, IMDB_BINARY, ENYZME):
        * load_data() -> imdb_dataset, cora_dataset, enzyme_dataset
    * Loading all 4 datasets( CORA, IMDB_BINARY, ENZYME, PEPTIDE-FUNC)
        * load_data_all() -> imdb_dataset, cora_dataset, enzyme_dataset, peptide_dataset
    * Load with Train-Test split (80-20 with shuffle):
        * load_imdb()
        * load_cora()
        * load_enzyme() 
        * load_peptides_func()
    * Running performance tests for GAT and GCN architectures based on layers:
        * mod_layers_GCN(dataset, layers, epochs=200) -> final test accuracy
        * mod_layers_GAT(dataset, layers, epochs=200) -> final test accuracy
     
### run.py
Runs the entirety of all the training and testing for ALL datasets on ALL models. You can run this on the command line if needed.
1.  `pip install -r requirements.txt` OR `conda env create -f environment.yml`
2.  `python3 run.py`

If you would prefer a step by step, you can look towards the notebooks for a more granular and more in detail usage of the code.
