from scipy.sparse import lil_matrix, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import os
from common import *
import pickle
from alive_progress import alive_bar


class GATv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
        super(GATv2, self).__init__()
        self.dropout = dropout

        # First GATv2 layer with edge_dim parameter
        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=1
        )

        # Second GATv2 layer with edge_dim parameter
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=1
        )

        # Output layer
        self.linear = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # Reshape edge_weight for GATv2Conv
        if edge_weight is not None:
            edge_weight = edge_weight.unsqueeze(-1)

        # First GAT layer
        # Change edge_weight to edge_attr
        x = self.conv1(x, edge_index, edge_attr=edge_weight)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GAT layer
        # Change edge_weight to edge_attr
        x = self.conv2(x, edge_index, edge_attr=edge_weight)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer
        x = self.linear(x)
        return x


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
    criterion = nn.MSELoss()

    # Split indices for train/validation
    num_nodes = data.x.size(0)
    indices = list(range(num_nodes))  # Convert to list for safer splitting
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42)

    # Convert to tensors but make sure they don't exceed the data size
    train_idx = torch.LongTensor([i for i in train_idx if i < num_nodes])
    val_idx = torch.LongTensor([i for i in val_idx if i < num_nodes])

    # Verify indices are within bounds
    assert max(train_idx) < num_nodes, "Training indices out of bounds"
    assert max(val_idx) < num_nodes, "Validation indices out of bounds"

    with alive_bar(num_epochs) as bar:
        for epoch in range(num_epochs):
            # Training
            model.train()
            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.edge_weight)

            # Make sure targets are properly aligned
            train_targets = data.y[train_idx]
            if len(train_targets.shape) == 1:
                train_targets = train_targets.unsqueeze(1)

            loss = criterion(out[train_idx], train_targets)

            loss.backward()
            optimizer.step()

            # Validation
            # model.eval()
            # with torch.no_grad():
            #     val_targets = data.y[val_idx]
            #     if len(val_targets.shape) == 1:
            #         val_targets = val_targets.unsqueeze(1)

            # val_loss = criterion(out[val_idx], val_targets)

            # Print progress
            # if epoch % 10 == 0:
            #     print(
            #         f'Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            bar()

    return model


def create_data(node_features, adjacency_matrix_sparse, weight_matrix_sparse, targets, device):
    data = process_data(node_features, adjacency_matrix_sparse,
                        weight_matrix_sparse, targets, device)
    return data


def create_model(node_features, device):
    model = GATv2(
        in_channels=node_features.shape[1],
        hidden_channels=128,
        out_channels=1,
        heads=TAU_MAX,
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

            # Map variables to their respective nodes in the flattened graph
            pos_i = time_lag * num_var + variable_i
            pos_j = time_lag * num_var + variable_j
            if link_type == '-->':
                adjacency_matrix[pos_i, pos_j] = 1
                weight_matrix[pos_i, pos_j] = link_value
            elif link_type == '<--':
                adjacency_matrix[pos_j, pos_i] = 1
                weight_matrix[pos_j, pos_i] = link_value
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
    action_df = pd.read_csv("fkeys/action.csv")
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


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    OVERWRITE_MODEL = False
    print(f'Using device: {device}')
    print("Loading data...")
    pcmci_links = pd.read_csv("links.csv")
    action_df = pd.read_csv("fkeys/action.csv")

    num_var = action_df['id'].max()+1

    print("Creating adjacency and weight matrix...")
    adjacency_matrix_sparse, weight_matrix_sparse = create_matrix(
        pcmci_links, num_var, TAU_MAX)

    print("Creating node features...")
    node_features = create_node_features(pcmci_links, num_var, TAU_MAX)

    print("Creating targets...")
    targets = create_targets(num_var, TAU_MAX)

    data = create_data(node_features, adjacency_matrix_sparse,
                       weight_matrix_sparse, targets, device)
    if (not os.path.exists("model.pkl")) or OVERWRITE_MODEL:
        print("Creating model...")
        model = create_model(node_features, device)
        print("Training model...")
        model = train(model, data, device)
        pickle.dump(model, open("model.pkl", "wb"))
    else:
        print("Loading model...")
        model = pickle.load(open("model.pkl", "rb"))

    print("Making predictions...")
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index, data.edge_weight)

        # Move tensors to CPU for metric calculation
        pred_np = predictions.cpu().squeeze().numpy()
        true_np = data.y.cpu().numpy()

        # Convert continuous values to binary using threshold of 0.5
        pred_np = (pred_np >= 0.5).astype(int)
        true_np = (true_np >= 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(true_np, pred_np)
        precision = precision_score(true_np, pred_np, zero_division=0)
        recall = recall_score(true_np, pred_np, zero_division=0)
        f1 = f1_score(true_np, pred_np, zero_division=0)
        print(
            f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
