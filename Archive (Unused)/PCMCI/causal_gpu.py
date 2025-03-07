# Imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import pickle
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sklearn

import tigramite
from tigramite import data_processing as pp
# from tigramite.toymodels import structural_causal_processes as toys

from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
# from tigramite.lpcmci import LPCMCI
from tigramite.independence_tests.parcorr import ParCorr
# from tigramite.independence_tests.robust_parcorr import RobustParCorr
# from tigramite.independence_tests.parcorr_wls import ParCorrWLS
# from tigramite.independence_tests.gpdc import GPDC
# from tigramite.independence_tests.cmiknn import CMIknn
# from tigramite.independence_tests.cmisymb import CMIsymb
# from tigramite.independence_tests.gsquared import Gsquared
# from tigramite.independence_tests.regressionCI import RegressionCI
from tigramite.models import LinearMediation, Prediction
import os

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import multiprocessing

print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device != torch.device('cuda'):
    raise ValueError("CUDA not available!")

TAU_MAX = 5
save_folder = 'save_6'
file_path = 'causal_data_5.csv'

if save_folder is not None:
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
# Define the GNN Model


class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)


def plot_and_save_graph(results, var_names, save_folder):
    tp.plot_graph(
        val_matrix=results['val_matrix'],
        graph=results['graph'],
        var_names=var_names,
        link_colorbar_label='cross-MCI',
        node_colorbar_label='auto-MCI',
        show_autodependency_lags=False,
        save_name=f'{save_folder}/graph.png' if save_folder is not None else None
    )
    plt.close()


def plot_and_save_time_series_graph(results, var_names, save_folder):
    tp.plot_time_series_graph(
        figsize=(6, 4),
        val_matrix=results['val_matrix'],
        graph=results['graph'],
        var_names=var_names,
        link_colorbar_label='MCI',
        save_name=f'{save_folder}/time_series_graph.png' if save_folder is not None else None
    )
    plt.close()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()


def save_model(model, optimizer, epoch, file_path=f"{save_folder}/gnn_model.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)
    print(f"Model saved to {file_path}")


def load_model(model, optimizer, file_path=f"{save_folder}/gnn_model.pth"):
    # Ensure model is loaded to the correct device
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {file_path}, starting from epoch {epoch}")
    return model, optimizer, epoch


def save_data(data, file_path=f"{save_folder}/gnn_data.pth"):
    torch.save(data, file_path)
    print(f"Data saved to {file_path}")


def load_data(file_path=f"{save_folder}/gnn_data.pth"):
    data = torch.load(file_path)
    print(f"Data loaded from {file_path}")
    return data


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)  # Data is already on GPU
        predictions = out.argmax(dim=1)
        accuracy = accuracy_score(
            data.y.cpu().numpy(), predictions.cpu().numpy())
        precision = precision_score(data.y.cpu().numpy(
        ), predictions.cpu().numpy(), average='weighted')
        recall = recall_score(data.y.cpu().numpy(),
                              predictions.cpu().numpy(), average='weighted')
        f1 = f1_score(data.y.cpu().numpy(),
                      predictions.cpu().numpy(), average='weighted')
    return accuracy, precision, recall, f1


if __name__ == '__main__':
    df = pd.read_csv(file_path)
    datatime = {0: df['time'].to_numpy()}
    df = df.drop('time', axis=1)
    # df = df.drop('jerseyNumber', axis=1)
    var_names = list(df.columns)
    tigramite_df = pp.DataFrame(
        df.to_numpy(), datatime=datatime, var_names=var_names)
    parcorr = ParCorr(significance='analytic')
    print('Constructing PCMCI')
    pcmci = PCMCI(dataframe=tigramite_df, cond_ind_test=parcorr, verbosity=1)
    if save_folder is not None:
        with open(f'{save_folder}/pcmci.pkl', 'wb') as f:
            pickle.dump(pcmci, f)
    print('PCMCI saved')

    print('Getting lagged dependencies')
    correlations = pcmci.get_lagged_dependencies(
        tau_max=TAU_MAX, val_only=True)['val_matrix']
    lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names': var_names,
                                                                            'x_base': 5, 'y_base': .5}, name=f'{save_folder}/lagfuncs.png' if save_folder is not None else None)
    print('Lagged dependencies saved')

    print('Running PCMCI')
    results = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=None, alpha_level=0.01)
    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=results['p_matrix'], tau_max=TAU_MAX, fdr_method='fdr_bh')
    pcmci.print_significant_links(
        p_matrix=q_matrix,
        val_matrix=results['val_matrix'],
        alpha_level=0.01)
    graph = pcmci.get_graph_from_pmatrix(p_matrix=q_matrix, alpha_level=0.01,
                                         tau_min=0, tau_max=TAU_MAX, link_assumptions=None)
    results['graph'] = graph

    if save_folder is not None:
        with open(f'{save_folder}/results.pkl', 'wb') as f:
            pickle.dump(results, f)
    print('PCMCI results saved')

    print('Plotting graph')

    process1 = multiprocessing.Process(
        target=plot_and_save_graph,
        args=(results, var_names, save_folder)
    )
    process2 = multiprocessing.Process(
        target=plot_and_save_time_series_graph,
        args=(results, var_names, save_folder)
    )

    # Start both processes
    process1.start()
    process2.start()

    # Wait for both processes to complete
    process1.join()
    process2.join()
    print('Graphs saved')

    print('Saving links')
    if save_folder is not None:
        tp.write_csv(
            val_matrix=results['val_matrix'],
            graph=results['graph'],
            var_names=var_names,
            save_name=f'{save_folder}/links.csv',
            digits=5,
        )
    print('Links saved')

    print('Running PCMCI prediction')
    T = list(tigramite_df.T.values())[0]
    pred = Prediction(dataframe=tigramite_df,
                      cond_ind_test=ParCorr(),  # CMIknn ParCorr
                      prediction_model=sklearn.linear_model.LinearRegression(),
                      #         prediction_model = sklearn.gaussian_process.GaussianProcessRegressor(),
                      # prediction_model = sklearn.neighbors.KNeighborsRegressor(),
                      data_transform=None,
                      train_indices=range(int(0.8*T)),
                      test_indices=range(int(0.9*T), T),
                      verbosity=1
                      )
    # target = var_names.index('behaviour')
    target = 3
    predictors = pred.get_predictors(
        selected_targets=[target],
        steps_ahead=1,
        tau_max=TAU_MAX,
        pc_alpha=None
    )
    pred.fit(target_predictors=predictors,
             selected_targets=[target],
             tau_max=TAU_MAX)

    predicted = pred.predict(target)
    true_data = pred.get_test_array(j=target)[0]
    print('Prediction completed')

    print('Plotting prediction')
    plt.scatter(true_data, predicted)
    plt.title(r"NMAE = %.2f" %
              (np.abs(true_data - predicted).mean()/true_data.std()))
    plt.plot(true_data, true_data, 'k-')
    plt.xlabel('True test data')
    plt.ylabel('Predicted test data')

    if save_folder is not None:
        plt.savefig(f'{save_folder}/prediction_nmae.png')
        with open(f'{save_folder}/pred.pkl', 'wb') as f:
            pickle.dump(pred, f)

    print('Prediction saved')

    print('Creating GNN')
    df = pd.read_csv(file_path)
    if save_folder is None:
        raise ValueError("Save folder not specified! No links saved.")
    links = pd.read_csv(f"{save_folder}/links.csv")

    if df.empty:
        raise ValueError("Dataset is empty!")
    if links.empty:
        raise ValueError("Links file is empty!")

    # Create node features (excluding only non-numeric and target column)
    node_features = df.drop(columns=['behaviour']).values
    target = df['behaviour'].values

    # Ensure node features and target are not empty
    if node_features.size == 0 or target.size == 0:
        raise ValueError("Node features or target are empty!")

    # Create edge list and weights
    edge_index = []  # Stores source and target node indices
    edge_weight = []  # Stores edge weights

    node_map = {jersey: idx for idx, jersey in enumerate(
        df['jerseyNumber'].unique())}

    # Process links and add edges
    for _, row in links.iterrows():
        if row['Variable i'] in node_map and row['Variable j'] in node_map:
            i = node_map[row['Variable i']]
            j = node_map[row['Variable j']]
            edge_index.append([i, j])
            edge_weight.append(row['Link value'])

    # If no edges exist, add self-loops with weight 0
    if not edge_index:
        print("No edges found, adding self-loops with weight 0.")
        edge_index = [[i, i] for i in range(len(node_map))]
        # Assign a weight of 0 for each self-loop
        edge_weight = [0.0] * len(node_map)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    nodes = torch.tensor(node_features, dtype=torch.float)
    target = torch.tensor(target, dtype=torch.long)

    # Create PyTorch Geometric Data object
    data = Data(x=nodes, edge_index=edge_index,
                edge_attr=edge_weight, y=target).to(device)

    # Initialize model, loss, and optimizer
    model = GNNModel(in_channels=nodes.shape[1], hidden_channels=16, out_channels=len(
        torch.unique(target))).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop

    # Train the model
    for epoch in range(200):
        loss = train()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
    print('Training completed')

    # Save the model
    save_model(model, optimizer, epoch)
    save_data(data)
    print('Model saved')

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate(model, data)
    print(
        f"Accuracy: {accuracy*100:.4f}, Precision: {precision*100:.4f}, Recall: {recall*100:.4f}, F1: {f1*100:.4f}")
    print('Evaluation completed')
