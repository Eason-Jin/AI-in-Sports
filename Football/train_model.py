from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mutual_info_score
from torch_geometric.data import Data
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import os
from common import *
import pickle
from alive_progress import alive_bar
from models.gatv2 import GATv2
from models.gcn import GCN
import argparse


def process_data(node_features, adjacency_matrix_sparse, weight_matrix_sparse, targets, device):
    # Convert to PyTorch tensors and move to GPU
    x = torch.FloatTensor(node_features).to(device)
    # Get edge indices from sparse adjacency matrix
    edge_index = torch.LongTensor(
        np.array(adjacency_matrix_sparse.nonzero())).to(device)

    # Get edge weights from weight matrix
    edge_weight = torch.FloatTensor(weight_matrix_sparse.data).to(device)

    # Convert targets
    y = torch.FloatTensor(targets).to(device)
    # Create PyTorch Geometric Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y
    )

    return data


def train(model, data, device, num_epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Split indices for train/validation
    num_nodes = data.x.size(0)
    indices = list(range(num_nodes))  # Convert to list for safer splitting
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42)

    with alive_bar(num_epochs) as bar:
        for epoch in range(num_epochs):
            # Training
            model.train()
            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.edge_weight)

            # Make sure targets are properly aligned
            train_targets = data.y[train_idx].long()
            # if len(train_targets.shape) == 1:
            #     train_targets = train_targets.unsqueeze(1)

            loss = criterion(out[train_idx], train_targets)

            loss.backward()
            optimizer.step()

            bar()

    return model


def create_data(node_features, adjacency_matrix_sparse, weight_matrix_sparse, targets, device):
    data = process_data(node_features, adjacency_matrix_sparse,
                        weight_matrix_sparse, targets, device)
    return data


def create_model(node_features, out_channels, device, model_type):
    if model_type == 'gatv2':
        model = GATv2(
            in_channels=node_features.shape[1],
            hidden_channels=128,
            out_channels=out_channels,
            heads=TAU_MAX,
            dropout=0.2
        ).to(device)
    elif model_type == 'gcn':
        model = GCN(
            in_channels=node_features.shape[1],
            hidden_channels=128,
            out_channels=out_channels,
            dropout=0.2
        ).to(device)

    return model


def create_matrix(pcmci_links, num_var, tau):
    # Time lag goes from 0 to TAU_MAX inclusive
    total_nodes = (tau+1) * num_var
    adjacency_matrix = lil_matrix((total_nodes, total_nodes), dtype=np.float32)
    weight_matrix = lil_matrix((total_nodes, total_nodes), dtype=np.float32)

    with alive_bar(len(pcmci_links)) as bar:
        for _, row in pcmci_links.iterrows():
            variable_i, variable_j, time_lag, link_type, link_value = read_pcmci_row(
                row)
            if link_value == 0:
                bar()
                continue
            # Map variables to their respective nodes in the flattened graph
            pos_i = time_lag * num_var + variable_i
            pos_j = time_lag * num_var + variable_j
            if link_type == '-->':
                adjacency_matrix[pos_i, pos_j] = 1
                weight_matrix[pos_i, pos_j] = link_value
            elif link_type == 'o-o':
                adjacency_matrix[pos_i, pos_j] = 1
                weight_matrix[pos_i, pos_j] = link_value
                adjacency_matrix[pos_j, pos_i] = 1
                weight_matrix[pos_j, pos_i] = link_value
            bar()

    # Convert to CSR format for efficient computation
    adjacency_matrix_sparse = adjacency_matrix.tocsr()
    weight_matrix_sparse = weight_matrix.tocsr()

    return adjacency_matrix_sparse, weight_matrix_sparse


def read_pcmci_row(row):
    global action_path
    action_df = pd.read_csv(action_path)
    variable_i = row["Variable i"]
    variable_i = int(
        action_df[action_df['action'] == variable_i]['id'].iloc[0])
    variable_j = row["Variable j"]
    variable_j = int(
        action_df[action_df['action'] == variable_j]['id'].iloc[0])
    time_lag = int(row["Time lag of i"])
    link_type = row["Link type i --- j"]
    link_value = float(row["Link value"])

    return variable_i, variable_j, time_lag, link_type, link_value


def create_node_features(pcmci_links, num_var, tau):
    """
    Node features should be a list of tau actions leading up to the node.
    """
    node_features = []
    with alive_bar(int((tau+1)*num_var)) as bar:
        # time = 0  (current)                [a1, a2, a3, ..., an], n=TAU_MAX
        # time = 1  (1 unit back in time)    [-1, a1, a2, a3, ..., an-1]
        # time = 2  (2 units back in time)   [-1, -1, a1, a2, ..., an-2]
        # ...
        # time = TAU_MAX (TAU_MAX units back in time) [-1, -1, -1, ..., -1]
        # Order of adjacency matrix is [time=0, time=1, time=2, ..., time=TAU_MAX]
        for t in range(tau):
            # Ignore the last one because it will be all -1
            for i in range(num_var):
                # Search in pcmci_links for all actions leads up to node (action i at time t)
                past_actions = [-1 for _ in range(tau)]
                time_cutoff = tau - t
                for _, row in pcmci_links.iterrows():
                    variable_i, variable_j, time_lag, _, _ = read_pcmci_row(
                        row)

                    if variable_j == i and time_lag <= time_cutoff and time_lag > 0:
                        index = tau - time_lag
                        if past_actions[index] == -1:
                            # TODO: Need a better solution, for now just use the first one
                            past_actions[index] = variable_i
                bar()
                node_features.append(past_actions)
        for _ in range(num_var):
            node_features.append([-1 for _ in range(tau)])
            bar()
    return np.array(node_features)


def create_targets(num_var, tau):
    """
    Each node (target) is an action at time lag, time lag goes from 0 to TAU_MAX.
    """
    targets = [i for i in range(num_var)] * (tau+1)
    targets = np.array(targets, dtype=np.float32)
    return targets


def evaluation(model, data):
    print("Making predictions...")
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index, data.edge_weight)

        # Move tensors to CPU for metric calculation
        pred_np = predictions.cpu().squeeze().numpy()
        true_np = data.y.cpu().numpy()

        pred_labels = np.argmax(pred_np, axis=1)

        # Convert true labels to integers
        true_labels = true_np.astype(int)

        print(true_labels)
        print(pred_labels)

        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(
            true_labels, pred_labels, zero_division=0, average='weighted')
        recall = recall_score(true_labels, pred_labels,
                              zero_division=0, average='weighted')
        f1 = f1_score(true_labels, pred_labels,
                      zero_division=0, average='weighted')
        mi = mutual_info_score(true_labels, pred_labels)
        print(
            f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, MI: {mi:.4f}')


if __name__ == '__main__':
    print(f"TAU_MAX: {TAU_MAX}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        help='Providing a model will only evaluate its performance.')
    parser.add_argument('--link-path', type=str, default="links.csv",
                        help='Path to the links CSV file (default: links.csv)')
    parser.add_argument('--action-path', type=str, default="fkeys/action.csv",
                        help='Path to the action CSV file (default: fkeys/action.csv)')
    parser.add_argument('--model-type', type=str, default="gatv2")
    parser.add_argument('--tau', type=int, default=TAU_MAX)
    parser.add_argument('--misc-path', type=str)
    args = parser.parse_args()

    global action_path
    action_path = args.action_path
    global link_path
    link_path = args.link_path
    model_type = args.model_type

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f'Using device: {device}')
    print("Loading data...")
    pcmci_links = pd.read_csv(link_path)
    action_df = pd.read_csv(action_path)
    os.makedirs("saves", exist_ok=True)

    num_var = action_df['id'].max()+1

    if args.misc_path:
        print(f"Loading matrices from {args.matrix_path}...")
        adjacency_matrix_sparse, weight_matrix_sparse = pickle.load(
            open(args.misc_path, "rb"))
        print(f"Loading node features from {args.node_features_path}...")
        node_features = pickle.load(open(args.node_features_path, "rb"))
        print(f"Loading targets from {args.targets_path}...")
        targets = pickle.load(open(args.targets_path, "rb"))
    else:
        print("Creating adjacency and weight matrix...")
        adjacency_matrix_sparse, weight_matrix_sparse = create_matrix(
            pcmci_links, num_var, TAU_MAX)
        pickle.dump((adjacency_matrix_sparse, weight_matrix_sparse),
                    open("saves/matrices.pkl", "wb"))

        print("Creating node features...")
        node_features = create_node_features(pcmci_links, num_var, TAU_MAX)
        pickle.dump(node_features, open("saves/node_features.pkl", "wb"))

        print("Creating targets...")
        targets = create_targets(num_var, TAU_MAX)
        pickle.dump(targets, open("saves/targets.pkl", "wb"))

    data = create_data(node_features, adjacency_matrix_sparse,
                       weight_matrix_sparse, targets, device)

    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        model = pickle.load(open(args.model_path, "rb"))
    else:
        print("Creating model...")
        model = create_model(node_features, num_var,
                             device, model_type.lower())
        print("Training model...")
        model = train(model, data, device)
        pickle.dump(model, open("saves/model.pkl", "wb"))

    evaluation(model, data)
