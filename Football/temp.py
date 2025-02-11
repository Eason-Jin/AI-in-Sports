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


class GATv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.2):
        super(GATv2, self).__init__()
        self.dropout = dropout

        # First GATv2 layer with edge_dim parameter
        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=1  # Add this parameter
        )

        # Second GATv2 layer with edge_dim parameter
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=1  # Add this parameter
        )

        # Output layer
        self.linear = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # Reshape edge_weight for GATv2Conv
        if edge_weight is not None:
            edge_weight = edge_weight.unsqueeze(-1)  # Add this line

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
    # x = torch.FloatTensor(node_features).to(device)
    x = torch.FloatTensor(node_features)

    # Get edge indices from sparse adjacency matrix
    # edge_index = torch.LongTensor(
    #     np.array(adjacency_matrix_sparse.nonzero())).to(device)
    edge_index = torch.LongTensor(
        np.array(adjacency_matrix_sparse.nonzero()))

    # Get edge weights from weight matrix
    # edge_weight = torch.FloatTensor(weight_matrix_sparse.data).to(device)
    edge_weight = torch.FloatTensor(weight_matrix_sparse.data)

    # Convert targets
    # y = torch.FloatTensor(targets).to(device)
    y = torch.FloatTensor(targets)

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

    best_val_loss = float('inf')
    patience = 10
    counter = 0

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
        model.eval()
        with torch.no_grad():
            val_targets = data.y[val_idx]
            if len(val_targets.shape) == 1:
                val_targets = val_targets.unsqueeze(1)

            val_loss = criterion(out[val_idx], val_targets)

        # Print progress
        if epoch % 10 == 0:
            print(
                f'Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        # # Early stopping
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print(f'Early stopping at epoch {epoch}')
        #         break

    return model


def create_model(node_features, adjacency_matrix_sparse, weight_matrix_sparse, targets, device):
    # Prepare data
    data = process_data(node_features, adjacency_matrix_sparse,
                        weight_matrix_sparse, targets, device)

    # Initialize model
    # model = GATv2(
    #     in_channels=node_features.shape[1],
    #     hidden_channels=32,
    #     out_channels=1,
    #     heads=4,
    #     dropout=0.2
    # ).to(device)
    model = GATv2(
        in_channels=node_features.shape[1],
        hidden_channels=32,
        out_channels=1,
        heads=4,
        dropout=0.2
    )

    return model, data

def create_matrix(pcmci_links, num_var, tau):
    """
    Flatten the 3D tensor into a 2D adjacency matrix.
    
    Args:
        pcmci_links (pd.DataFrame): The PCMCI links dataframe.
        num_var (int): Number of actions (nodes).
        tau (int): Maximum time lag.
    
    Returns:
        adjacency_matrix (scipy.sparse.csr_matrix): Flattened 2D adjacency matrix.
        weight_matrix (scipy.sparse.csr_matrix): Flattened 2D weight matrix.
    """
    action_df = pd.read_csv("fkeys/action.csv")
    total_nodes = tau * num_var
    adjacency_matrix = lil_matrix((total_nodes, total_nodes), dtype=np.float32)
    weight_matrix = lil_matrix((total_nodes, total_nodes), dtype=np.float32)

    for _, row in pcmci_links.iterrows():
        variable_i = row["Variable i"]
        variable_i = int(action_df[action_df['action'] == variable_i]['id'])
        variable_j = row["Variable j"]
        variable_j = int(action_df[action_df['action'] == variable_j]['id'])
        source_time_lag = int(row["Time lag of i"])
        link_value = float(row["Link value"])

        # Map variables to their respective nodes in the flattened graph
        for t in range(tau):
            if t - source_time_lag >= 0:
                # Source node: variable_i at time (t - source_time_lag)
                source_node = (t - source_time_lag) * num_var + variable_i
                # Target node: variable_j at time t
                target_node = t * num_var + variable_j
                adjacency_matrix[source_node, target_node] = 1
                weight_matrix[source_node, target_node] = link_value

    # Convert to CSR format for efficient computation
    adjacency_matrix_sparse = adjacency_matrix.tocsr()
    weight_matrix_sparse = weight_matrix.tocsr()

    return adjacency_matrix_sparse, weight_matrix_sparse

def create_node_features(num_var, tau):
    """
    Create node features for the flattened graph.
    
    Args:
        num_var (int): Number of actions (nodes).
        tau (int): Maximum time lag.
    
    Returns:
        node_features (np.array): Node features for the flattened graph.
    """
    total_nodes = tau * num_var
    node_features = np.zeros((total_nodes, num_var + 1), dtype=np.float32)

    for t in range(tau):
        for a in range(num_var):
            node_idx = t * num_var + a
            # One-hot encode the action
            node_features[node_idx, a] = 1
            # Add time step as a feature
            node_features[node_idx, num_var] = t

    return node_features

def create_targets(actions, num_var, tau):
    """
    Create targets for the flattened graph.
    
    Args:
        actions (np.array): Array of actions over time.
        num_var (int): Number of actions (nodes).
        tau (int): Maximum time lag.
    
    Returns:
        targets (np.array): Targets for the flattened graph.
    """
    total_nodes = tau * num_var
    targets = np.zeros(total_nodes, dtype=np.float32)

    for t in range(tau - 1):
        for a in range(num_var):
            node_idx = t * num_var + a
            # Target is the action at the next time step
            targets[node_idx] = actions[t + 1, a]

    return targets

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    OVERWRITE_MATRIX = True
    OVERWRITE_MODEL = True
    print(f'Using device: {device}')
    print("Loading data...")
    # Load and concatenate data
    df = pd.DataFrame()
    for folder in os.listdir("matches"):
        if os.path.exists(f"matches/{folder}/match_data_gat.csv"):
            df = pd.concat(
                [df, pd.read_csv(f"matches/{folder}/match_data_gat.csv")])

    pcmci_links = pd.read_csv("links.csv")
    time_steps_per_block = int(df["time"].max() + 1)
    num_blocks = int(len(df) // time_steps_per_block)

    actions = [df["action"].values[i *
                                   time_steps_per_block:(i + 1) * time_steps_per_block] for i in range(num_blocks)]
    actions = np.array(actions)

    num_var = len(df['action'].unique())

    if (not os.path.exists("adjacency_matrix_sparse.pkl") and not os.path.exists("weight_matrix_sparse.pkl")) or OVERWRITE_MATRIX:
        print("Creating adjacency and weight matrix...")
        adjacency_matrix_sparse, weight_matrix_sparse = create_matrix(pcmci_links, num_var, TAU_MAX)

        pickle.dump(adjacency_matrix_sparse, open(
            "adjacency_matrix_sparse.pkl", "wb"))
        pickle.dump(weight_matrix_sparse, open(
            "weight_matrix_sparse.pkl", "wb"))
    else:
        print("Loading adjacency and weight matrix...")
        adjacency_matrix_sparse = pickle.load(
            open("adjacency_matrix_sparse.pkl", "rb"))
        weight_matrix_sparse = pickle.load(
            open("weight_matrix_sparse.pkl", "rb"))

    # Step 5: Prepare targets (future actions within the same block)
    print("Creating targets...")
    targets = create_targets(actions, num_var, TAU_MAX)
    node_features = create_node_features(num_var, TAU_MAX)

    if (not os.path.exists("model.pkl")) or OVERWRITE_MODEL:
        print("Creating model...")
        model, data = create_model(node_features, adjacency_matrix_sparse,
                                weight_matrix_sparse, targets, device)
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
        # pred_np = predictions.cpu().squeeze().numpy()
        pred_np = predictions.squeeze().numpy()
        # true_np = data.y.cpu().numpy()
        true_np = data.y.numpy()

        # Convert continuous values to binary using threshold of 0.5
        pred_np = (pred_np >= 0.5).astype(int)
        true_np = (true_np >= 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(true_np, pred_np)
        precision = precision_score(true_np, pred_np, zero_division=0)
        recall = recall_score(true_np, pred_np, zero_division=0)
        f1 = f1_score(true_np, pred_np, zero_division=0)
        print(
            f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}')
