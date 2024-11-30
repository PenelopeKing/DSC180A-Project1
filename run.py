# contains the entire pipeline
import sys
import os
import random

# Add the absolute path to the src directory 
#os.chdir('src/')
src_path = os.path.abspath('src/')
sys.path.insert(0, src_path)

from setup import *
seed = 123
import random
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def main():
    """runs the training and testing for ALL DATASETS and ALL MODELS, as shown in notebooks"""
    print('LOADING DATASETS...')
    imdb_dataset, cora_dataset, enzyme_dataset = load_data()

    # CORA # 
    print('\nBEGINNING CORA')
    print('LOADING CORA DATA...')

    print('\nGCN')
    # init model and optimizer
    hidden_channels = 32
    layers = 2
    cora_mdl = GCNNode(cora_dataset.num_features, 
                        hidden_channels, cora_dataset.num_classes, 
                        num_layers= layers)
    optimizer = torch.optim.Adam(cora_mdl.parameters(), lr=0.01, weight_decay=5e-4)

    # train model
    for _ in range(200):
        cora_mdl = node_train(model = cora_mdl, data = cora_dataset, optimizer=optimizer)
    # calculate accuracy
    cora_test_acc, cora_train_acc, pred = node_test(cora_mdl, cora_dataset)
    print(f'CORA Test Acc: {cora_test_acc:.4f}')
    print(f'CORA Train Acc: {cora_train_acc:.4f}')

    print('\nGIN')
    # init model
    cora_mdl = GINNode(in_channels = cora_dataset.num_features,
                hidden_channels = 30,
                out_channels = cora_dataset.num_classes,
                num_layers = 4)
    optimizer = torch.optim.Adam(cora_mdl.parameters(), lr=0.01)

    # train model
    for epoch in range(500):
        cora_mdl = node_train(model = cora_mdl, 
                                    data = cora_dataset, 
                                    optimizer=optimizer)

    # evaluation
    test_acc, train_acc, pred = node_test(cora_mdl, cora_dataset)
    print(f'CORA GIN Test Accuracy: {test_acc:.4f}')
    print(f'CORA GIN Train Accuracy: {train_acc:.4f}')

    print('\nGAT')
    # init model
    hidden_channels = 32 
    layers = 2
    cora_mdl = GATNode(cora_dataset.num_features, 
                        hidden_channels, 
                        cora_dataset.num_classes, 16, layers)
    optimizer = torch.optim.Adam(cora_mdl.parameters(), 
                                lr=0.005, 
                                weight_decay=5e-4)

    # train and test model
    for _ in range(100):
        cora_mdl = node_train(cora_mdl, cora_dataset, optimizer)

    # calculate accuracy
    cora_test_acc, cora_train_acc, pred = node_test(cora_mdl, cora_dataset)
    print(f'CORA Test Acc: {cora_test_acc:.4f}')
    print(f'CORA Train Acc: {cora_train_acc:.4f}')

    print('\nGraphGPS')
    # Load data
    cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')[0]
    num_node_features = cora_dataset.num_node_features
    num_classes = int(cora_dataset.y.max().item()) + 1  # Assuming labels are from 0 to num_classes - 1

    # init vars and params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPSNode(
        num_node_features=num_node_features,
        hidden_channels=132,
        num_classes=num_classes,
        num_layers=8,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-4)

    # training loop, get train test acc
    for epoch in range(80):
        loss = train_gps_nodes(model, cora_dataset, optimizer, device)
        train_acc, test_acc = test_gps_nodes(model, cora_dataset, device)
        scheduler.step(loss)

    print(f'Final Loss: {loss:.4f}, Final Train Acc: {train_acc:.4f} , Final Test Acc: {test_acc:.4f}')

    # IMDB # 
    print('\nBEGINNING IMDB-BINARY')
    print('LOADING IMDB-BINARY DATA...')

    print('\nGCN')
    train_loader, test_loader = preprocess_data(imdb_dataset, onehot=True, batch_size=64)
    # train model
    imdb_mdl = GCNGraph(hidden_channels=64, num_classes = imdb_dataset.num_classes, 
                        num_node_features = imdb_dataset.num_node_features, layers = 2)
    optimizer = torch.optim.Adam(imdb_mdl.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(200):
        graph_train(imdb_mdl, train_loader, optimizer, criterion)
    test_acc, train_acc = graph_test(imdb_mdl, test_loader, train_loader)
    print(f"Final TEST Accuracy on IMDB: {test_acc:.4f}")
    print(f"Final TRAIN Accuracy on IMDB: {train_acc:.4f}")


    print('\nGIN')
    # init model
    imdb_model = GINGraph(in_channels=imdb_dataset.num_node_features,
                    hidden_channels=32, 
                    out_channels=imdb_dataset.num_classes,
                    num_layers = 4)
    optimizer = torch.optim.Adam(imdb_model.parameters(), 
                                lr=0.01)

    # Training the model
    for epoch in range(200): 
        train_loss = graph_train(imdb_model, train_loader, optimizer)
    # evaluate
    test_acc, train_acc = graph_test(imdb_model, test_loader, train_loader)
    print(f"Final TEST Accuracy on IMDB_BINARY: {test_acc:.4f}")
    print(f"Final TRAIN Accuracy on IMDB_BINARY: {train_acc:.4f}")
    
    print('\nGAT')
    # init model
    hidden_channels = 15
    layers = 2
    heads = 5
    imdb_mdl = GATGraph(imdb_dataset.num_features, 
                            hidden_channels, 
                            imdb_dataset.num_classes, heads, layers)
    optimizer = torch.optim.Adam(imdb_mdl.parameters(), lr=0.01)
    # train model
    for _ in range(100):
        graph_train(imdb_mdl, train_loader, optimizer)
    # test and trian accuracy
    test_acc, train_acc = graph_test(imdb_mdl, test_loader, train_loader)
    print(f"Final TEST Accuracy on IMDB: {test_acc:.4f}")
    print(f"Final TRAIN Accuracy on IMDB': {train_acc:.4f}")

    print('\nGraphGPS')
    # load data
    imdb_train_loader, imdb_test_loader = load_imdb()
    imdb_dataset = imdb_train_loader.dataset
    num_node_features = imdb_dataset[0].num_node_features

    # input params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn_kwargs = {'dropout': 0.5}
    model = GPSGraph(channels=64,\
                num_node_features = num_node_features, \
                    num_layers=4, attn_type='performer', \
                        attn_kwargs=attn_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001)

    # train test acc
    for epoch in range(80):
        loss = train_gps_graph(model, imdb_train_loader, optimizer, device)
        test_acc = test_gps_graph(model, imdb_test_loader, device)
        train_acc = test_gps_graph(model, imdb_train_loader, device)
        scheduler.step(loss)

    print(f'Final Loss: {loss:.4f}, Final Train Acc: {train_acc:.4f}, Final Test Acc: {test_acc:.4f}')



    # ENZYME # 
    print('\nBEGINNING ENZYME')

    print('\nGCN')
    # set up dataset - split train/test data loaders
    train_loader, test_loader = preprocess_data(enzyme_dataset, onehot=False, batch_size=64)
    # train model
    hidden_channels = 32
    enzyme_mdl = GCNGraph(hidden_channels = hidden_channels,
                        num_node_features = enzyme_dataset.num_node_features,
                        num_classes= enzyme_dataset.num_classes, layers = 2)
    optimizer = torch.optim.Adam(enzyme_mdl.parameters(), lr=0.01)

    for epoch in range(200):
        graph_train(enzyme_mdl, train_loader, optimizer, criterion)

    test_acc, train_acc = graph_test(enzyme_mdl, test_loader, train_loader)
    print(f"Final TEST Accuracy on ENZYME: {test_acc:.4f}")
    print(f"Final TRAIN Accuracy on ENZYME: {train_acc:.4f}")


    print('\nGIN')
    # init model
    enzyme_mdl = GINGraph(enzyme_dataset.num_features,
                            80,
                            enzyme_dataset.num_classes,
                            num_layers=4)
    #enzyme_mdl = torch.jit.script(enzyme_mdl)
    optimizer = torch.optim.Adam(enzyme_mdl.parameters(),
                                lr=0.001,
                                weight_decay=1e-5)

    # train model
    for epoch in range(100):
        loss = graph_train(enzyme_mdl, train_loader, optimizer)

    # evaluate on test data
    test_acc, train_acc = graph_test(enzyme_mdl, test_loader, train_loader)
    print(f"Final TEST Accuracy on ENZYME: {test_acc:.4f}")
    print(f"Final TRAIN Accuracy on ENZYME: {train_acc:.4f}")


    print('\nGAT')
    # init model
    hidden_channels = 132
    layers = 5
    heads = 2
    enzyme_mdl = GATGraph(enzyme_dataset.num_features, 
                            hidden_channels, 
                            enzyme_dataset.num_classes,
                            heads, layers)
    optimizer = torch.optim.Adam(enzyme_mdl.parameters(), lr=0.000001,
                                weight_decay=1e-5)

    # train model
    for epoch in range(100):
        enzyme_mdl = graph_train(enzyme_mdl, train_loader, optimizer)
        test_acc, train_acc = graph_test(enzyme_mdl, test_loader, train_loader)
        if epoch % 10 == 0:
            print(f"EPOCH {epoch} : Train Acc = {train_acc:.4f} , Test Acc = {test_acc:.4f}")

    # get test and train acc
    test_acc, train_acc = graph_test(enzyme_mdl, test_loader, train_loader)
    print(f"Final TEST Accuracy on ENZYME: {test_acc:.4f}")
    print(f"Final TRAIN Accuracy on ENZYME: {train_acc:.4f}")

    print('\nGraphGPS')
    # load data
    enzyme_train_loader, enzyme_test_loader = load_enzyme()
    enzyme_dataset = enzyme_train_loader.dataset
    num_node_features = enzyme_dataset[0].num_node_features

    # input params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn_kwargs = {'dropout': 0.5}
    model = GPSGraph(channels=132,\
                num_node_features = num_node_features, \
                    num_layers=8, attn_type='performer', \
                        attn_kwargs=attn_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00000001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.0001)

    # train test acc
    for epoch in range(80):
        loss = train_gps_graph(model, enzyme_train_loader, optimizer, device)
        test_acc = test_gps_graph(model, enzyme_train_loader, device)
        train_acc = test_gps_graph(model, enzyme_test_loader, device)
        scheduler.step(loss)

    print(f'Final Loss: {loss:.4f},  Final Train Acc: {train_acc:.4f}, Final Test Acc: {test_acc:.4f}')


    # PEPTIDE-FUNC (long range) # 
    # load the data
    print('\nBEGINNING LONG RANGE DATASET (PEPTIDE-FUNC)')
    mp.set_start_method('spawn', force=True)
    print('LOADING PEPTIDE-FUNC DATA...')
    # Load entire Peptides-func dataset
    train_loader, test_loader, num_classes = load_peptides_func(parallel=True, subset_ratio=1.0)
    num_node_features = 9 

    print('\nGCN')
    # Define model, optimizer, and scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNGraph(
        num_node_features=num_node_features,
        hidden_channels=32,
        num_classes=num_classes,
        layers=3
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print('Starting training')

    # Training model, print final results
    for epoch in range(80):
        train_loss = longrange_train(model, train_loader, optimizer, device)
        train_acc = longrange_test(model, train_loader, device)
        test_acc = longrange_test(model, test_loader, device)
        if epoch % 10 == 0:
            print(f"EPOCH {epoch} : Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} , Test Acc = {test_acc:.4f}")
            
        scheduler.step(train_loss)
    print(f"FINAL: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} , Test Acc = {test_acc:.4f}")

    print('\nGIN')
    # Define model, optimizer, and scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GINGraph(
        in_channels=num_node_features,
        hidden_channels=20,
    out_channels=num_classes,
    num_layers=3
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print('Starting training')

    # Training model, print final results
    for epoch in range(50):
        train_loss = longrange_train(model, train_loader, optimizer, device)
        train_acc = longrange_test(model, train_loader, device)
        test_acc = longrange_test(model, test_loader, device)
        if epoch % 10 == 0:
            print(f"EPOCH {epoch} : Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} , Test Acc = {test_acc:.4f}")
        scheduler.step(train_loss)
    print(f"FINAL: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} , Test Acc = {test_acc:.4f}")
        
    print('\nGAT')
    # Define model, optimizer, and scheduler
    model = GATGraph(
        in_channels=num_node_features,
        hidden_channels=32,
        out_channels=num_classes,
        heads = 2,
        layers= 3
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print('Starting training')

    # Training model, print final results
    for epoch in range(80):
        train_loss = longrange_train(model, train_loader, optimizer, device)
        train_acc = longrange_test(model, train_loader, device)
        test_acc = longrange_test(model, test_loader, device)
        if epoch % 10 == 0:
            print(f"EPOCH {epoch} : Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} , Test Acc = {test_acc:.4f}")
        scheduler.step(train_loss)
        
    print(f"FINAL: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} , Test Acc = {test_acc:.4f}")

    print('\nGraphGPS')
    # Define model, optimizer, and scheduler
    model = GPSLongRange(
        num_node_features=num_node_features,
        hidden_channels=32,
        num_classes=num_classes,
        num_layers=5
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print('Starting training')

    # Training model, print final results
    for epoch in range(150):
        train_loss = longrange_train(model, train_loader, optimizer, device)
        train_acc = longrange_test(model, train_loader, device)
        test_acc = longrange_test(model, test_loader, device)
        if epoch % 10 == 0:
            print(f"EPOCH {epoch} : Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} , Test Acc = {test_acc:.4f}")
            
        scheduler.step(train_loss)
    print(f"FINAL: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} , Test Acc = {test_acc:.4f}")
    print('\nDONE!')


if __name__=="__main__":
    main()


