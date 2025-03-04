from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mutual_info_score
from torch_geometric.data import Data
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from common import *
import pickle
from alive_progress import alive_bar
from models.gatv2 import GATv2
from models.gcn import GCN
import argparse


def create_data(node_features, adjacency_matrix_sparse, weight_matrix_sparse, targets, device):
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
    # tau = 5
    print(x.shape)              # 110, 5 = num_var*tau, tau
    print(y.shape)              # 110    = num_var*tau
    print(edge_index.shape)     # 2, 250
    print(edge_weight.shape)    # 250    = num_edges (ignored tau 0)
    return data


def create_model(node_features, out_channels, device, model_type):
    print(f"Creating model of type {model_type}...")
    if model_type == ModelTypes.GATv2:
        model = GATv2(
            in_channels=node_features.shape[1],
            hidden_channels=128,
            out_channels=out_channels,
            heads=tau_max,
            dropout=0.2
        ).to(device)
    elif model_type == ModelTypes.GCN:
        model = GCN(
            in_channels=node_features.shape[1],
            hidden_channels=128,
            out_channels=out_channels,
            dropout=0.2
        ).to(device)
    else:
        raise Exception("Model type not supported!")

    return model


def train_model(model, train_data, device, num_epochs=100, lr=0.001):
    print("Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    with alive_bar(num_epochs) as bar:
        for epoch in range(num_epochs):
            # Training
            model.train()
            optimizer.zero_grad()

            out = model(train_data.x, train_data.edge_index,
                        train_data.edge_weight)

            loss = criterion(out, train_data.y.long())

            loss.backward()
            optimizer.step()

            bar()

    return model


def create_matrices(pcmci_links, num_var, tau, link_type):
    print("Creating adjacency and weight matrix...")
    # Time lag goes from 0 to tau_max inclusive
    total_nodes = tau * num_var
    adjacency_matrix = lil_matrix((total_nodes, total_nodes), dtype=np.float32)
    weight_matrix = lil_matrix((total_nodes, total_nodes), dtype=np.float32)

    with alive_bar(len(pcmci_links)) as bar:
        for _, row in pcmci_links.iterrows():
            variable_i, variable_j, time_lag, link_type, link_value = read_pcmci_row(
                row)
            if link_value == 0 or time_lag == 0:
                bar()
                continue
            # Map variables to their respective nodes in the flattened graph
            pos_i = (time_lag-1) * num_var + variable_i
            pos_j = (time_lag-1) * num_var + variable_j
            random_number = np.random.rand()/1000
            l_v = random_number
            # if link_type == LinkTypes.RANDOM:
            #     random_number = np.random.rand()/1000
            #     l_v = random_number
            # elif link_type == LinkTypes.PCMCI:
            #     l_v = link_value
            # elif link_type == LinkTypes.INVERSE_COUNT:
            #     df = readMatchData("match_data.csv")
            #     counts = df['action'].value_counts().to_dict()
            #     total = sum(counts.values())
            #     try:
            #         l_v = 1 - counts[variable_i]/total
            #     except KeyError:
            #         l_v = 0
            # else:
            #     raise Exception("Link weight type not supported!")

            if link_type == '-->':
                adjacency_matrix[pos_i, pos_j] = 1
                weight_matrix[pos_i, pos_j] = l_v
            elif link_type == 'o-o':
                adjacency_matrix[pos_i, pos_j] = 1
                weight_matrix[pos_i, pos_j] = l_v
                adjacency_matrix[pos_j, pos_i] = 1
                weight_matrix[pos_j, pos_i] = l_v
            bar()

    # Convert to CSR format for efficient computation
    adjacency_matrix_sparse = adjacency_matrix.tocsr()
    weight_matrix_sparse = weight_matrix.tocsr()

    return adjacency_matrix_sparse, weight_matrix_sparse


def read_pcmci_row(row):
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

def create_node_features_2(player_df, num_var, tau):
    # player_df is a match_df subset with that player
    node_features = []
    # Each node feature has size tau and is a list of actions leading to current action
    for t in range(tau-1):
        for i in range(num_var):
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
            node_features.append(past_actions)
    for _ in range(num_var):
        node_features.append([-1 for _ in range(tau)])
    # print(node_features)
    return np.array(node_features)

def create_node_features(pcmci_links, num_var, tau):
    print("Creating node features...")
    """
    Node features should be a list of tau actions leading up to the node.
    """
    node_features = []
    with alive_bar(int(tau*num_var)) as bar:
        # time = 0  (current)                [a1, a2, a3, ..., an], n=tau_max
        # time = 1  (1 unit back in time)    [-1, a1, a2, a3, ..., an-1]
        # time = 2  (2 units back in time)   [-1, -1, a1, a2, ..., an-2]
        # ...
        # time = tau_max (tau_max units back in time) [-1, -1, -1, ..., -1]
        # Order of adjacency matrix is [time=0, time=1, time=2, ..., time=tau_max]
        for t in range(tau-1):
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
                # if args.fill:
                #     # Fill in the blanks with the previous action
                #     for i in range(1, len(past_actions)):
                #         if past_actions[i] == -1 and past_actions[i-1] != -1:
                #             past_actions[i] = past_actions[i-1]
                node_features.append(past_actions)
        for _ in range(num_var):
            node_features.append([-1 for _ in range(tau)])
            bar()
    # print(node_features)
    return np.array(node_features)


def create_targets(num_var, tau):
    print("Creating targets...")
    """
    Each node (target) is an action at time lag, time lag goes from 0 to tau_max.
    """
    targets = [i for i in range(num_var)] * tau
    targets = np.array(targets, dtype=np.float32)
    return targets


def evaluate_model(model, test_data):
    print("Making predictions...")
    model.eval()
    with torch.no_grad():
        predictions = model(
            test_data.x, test_data.edge_index, test_data.edge_weight)

        # Move tensors to CPU for metric calculation
        pred_np = predictions.cpu().squeeze().numpy()
        true_np = test_data.y.cpu().numpy()

        pred_labels = np.argmax(pred_np, axis=1)

        # Convert true labels to integers
        true_labels = true_np.astype(int)

        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(
            true_labels, pred_labels, zero_division=0, average='macro')
        recall = recall_score(true_labels, pred_labels,
                              zero_division=0, average='macro')
        f1 = f1_score(true_labels, pred_labels,
                      zero_division=0, average='macro')
        mi = mutual_info_score(true_labels, pred_labels)
        metrics = f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, MI: {mi:.4f}'
        print(metrics)
        if load_path is None:
            with open(f'{pcmci_path}/config.txt', 'w') as f:
                f.write(f'Model Type: {model_type}\n')
                f.write(f'Tau: {tau_max}\n')
                f.write(f'PCMCI Path: {pcmci_path}\n')
                f.write(metrics)
                f.write('\nNotes: ')
                if args.notes:
                    f.write(args.notes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pcmci-path', type=str,
    #                     default=get_last_subfolder("saves"))
    parser.add_argument('--pcmci-path', type=str,
                        default="saves/2025-02-19-14-54-36")
    parser.add_argument('--model-type', type=str,
                        default="gatv2", help=f"Possible model types: {[e.value for e in ModelTypes]}")
    parser.add_argument('--tau', type=int, default=TAU_MAX)
    parser.add_argument('--load-model', type=str,
                        help="Path to load existing model for evaluation")
    parser.add_argument(
        '--notes', type=str, help="Notes for the model (will be written to config.txt)")
    parser.add_argument('--link-type', type=str, default='random',
                        help=f'Possible link types: {[e.value for e in LinkTypes]}')
    args = parser.parse_args()
    pcmci_path = args.pcmci_path
    model_type = args.model_type
    tau_max = args.tau
    load_path = get_enum_key(ModelTypes, args.load_model)
    link_type = get_enum_key(LinkTypes, args.link_type)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f'Using device: {device}')
    print(f"Loading PCMCI and action data from {pcmci_path}...")
    pcmci_links = pd.read_csv(f'{pcmci_path}/links.csv')
    action_df = pd.read_csv(f'{pcmci_path}/action.csv')

    num_var = action_df['id'].max()+1

    if load_path:
        print(f"Loading matrices...")
        adjacency_matrix_sparse, weight_matrix_sparse = pickle.load(
            open(f'{load_path}/matrices.pkl', "rb"))
        print(f"Loading node features...")
        node_features = pickle.load(
            open(f'{load_path}/node_features.pkl', "rb"))
        print(f"Loading targets...")
        targets = pickle.load(open(f'{load_path}/targets.pkl', "rb"))
    else:
        adjacency_matrix_sparse, weight_matrix_sparse = create_matrices(
            pcmci_links, num_var, tau_max, link_type)
        pickle.dump((adjacency_matrix_sparse, weight_matrix_sparse),
                    open(f"{pcmci_path}/matrices.pkl", "wb"))

        node_features = create_node_features(pcmci_links, num_var, tau_max)
        pickle.dump(node_features, open(
            f"{pcmci_path}/node_features.pkl", "wb"))

        targets = create_targets(num_var, tau_max)
        pickle.dump(targets, open(f"{pcmci_path}/targets.pkl", "wb"))

    data = create_data(node_features, adjacency_matrix_sparse,
                       weight_matrix_sparse, targets, device)

    num_nodes = data.x.shape[0]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_indices, test_indices = train_test_split(
        np.arange(num_nodes), test_size=0.2)
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    train_data = Data(
        x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight,
        y=data.y, train_mask=train_mask, test_mask=None
    )

    test_data = Data(
        x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight,
        y=data.y, train_mask=None, test_mask=test_mask
    )

    if load_path:
        print(f"Loading model from {load_path}...")
        model = pickle.load(open(f'{load_path}/model.pkl', "rb"))
    else:
        model = create_model(node_features, num_var,
                             device, model_type.lower())
        model = train_model(model, train_data, device)
        pickle.dump((model), open(f"{pcmci_path}/model.pkl", "wb"))

    evaluate_model(model, test_data)
