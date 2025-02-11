import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dense_to_sparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
from common import *
from sklearn.neighbors import kneighbors_graph
from strong_ai.behaviour_model import *
from torch.utils.data import DataLoader
from strong_ai.dataset import SeriesDataset
from strong_ai.formatter import *
from strong_ai.evaluation import *
# from visualisation import *
import pickle


match_folders = [f for f in os.listdir(
    'matches') if os.path.isdir(os.path.join('matches', f))]

# df_list = []
# for match_folder in match_folders:
#     if os.path.exists(f'matches/{match_folder}/match_data_gat.csv'):
#         print(f'Processing {match_folder}')
#         df_list.append(pd.read_csv(
#             f'matches/{match_folder}/match_data_gat.csv'))
# df = pd.concat(df_list, ignore_index=True)

# match_data = [pd.read_csv(
#     f'matches/{name}/match_data_gat.csv') for name in match_folders]
match_data = []
for name in match_folders:
    try:
        match_data.append(pd.read_csv(
            f'matches/{name}/match_data_gat.csv'))
    except FileNotFoundError:
        print(f"Data up to {name}")
        break
    
assert len(match_data) > 1, "Not enough data"
test_data, train_data = train_test_split(match_data, test_size=0.2)

# test_formatter = PandasFormatterEnsemble(test_data)
# variables = test_formatter.get_formatted_columns()

# train_formatter = PandasFormatterEnsemble(train_data)
train_sequences, *_ = train_formatter.format(event_driven=True)
train_sequences = {i: sequence for i, sequence in enumerate(train_sequences)}


train_dataset = SeriesDataset(train_sequences, lookback=TAU_MAX+1)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

graph = pickle.load(open('graph.pkl', 'rb'))
val_matrix = pickle.load(open('val_matrix.pkl', 'rb'))

val_matrix = torch.nan_to_num(torch.from_numpy(val_matrix).float())
graph[np.where(graph != "-->")] = "0"
graph[np.where(graph == "-->")] = "1"
graph = graph.astype(np.int64)
graph = torch.from_numpy(graph).float()

weights = graph*val_matrix

num_var = 22
model = CausalGATv2Wrapper(num_var=num_var, lookback=TAU_MAX+1,
                           weights=weights)

trainer = pl.Trainer(
    max_epochs=10,
    devices=[0],
    accelerator="gpu")

trainer.fit(model, train_loader)

formatter = PandasFormatterEnsemble(match_data)
sequences, true_ind_sequences, neighbor_graphs, * \
    _ = formatter.format(event_driven=True)
sequences = {i: sequence for i, sequence in enumerate(sequences)}
variables = formatter.get_formatted_columns()
dataset = SeriesDataset(sequences, lookback=TAU_MAX+1)
random_loader = DataLoader(dataset, batch_size=4, shuffle=True)
model.eval()
with torch.no_grad():
    # Compute direct prediction accuracy
    acc, acc_last = direct_prediction_accuracy(model, random_loader, num_var)
    print(f"Direct Prediction Accuracy: {acc}")
    print(f"Direct Prediction Accuracy (last layer only): {acc_last}")

    # Compute conditional mutual information
    cmi = mutual_information(model, random_loader, num_var)
    print(f"Mutual Information: {cmi}")

    # Compute series prediction metrics
    series = generate_series(model, dataset, num_var)
    nb_series = len(series)
    print(f"Generated {nb_series} series.")

    MIN_LENGTH = 30
    series = {k: v for k, v in series.items() if len(v) >= MIN_LENGTH}
    print(
        f"Removed {nb_series - len(series)}/{nb_series} series with length < {MIN_LENGTH}.")


# Accuracy: 0.5560, Precision: 0.4025, Recall: 0.5560, F1-score: 0.4362 (with weight adjustment)
# Accuracy: 0.5630, Precision: 0.4324, Recall: 0.5630, F1-score: 0.4412 (no weight adjustment)
