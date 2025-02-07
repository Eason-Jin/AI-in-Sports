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
from model import *


match_folders = [f for f in os.listdir(
    'matches') if os.path.isdir(os.path.join('matches', f))]
df_list = []
for match_folder in match_folders:
    if os.path.exists(f'matches/{match_folder}/match_data_gat.csv'):
        print(f'Processing {match_folder}')
        df_list.append(pd.read_csv(
            f'matches/{match_folder}/match_data_gat.csv'))
df = pd.concat(df_list, ignore_index=True)


# Split into train and test sets
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]  # Target action

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)


# Initialize the model
num_features = X_train.shape[1]
num_classes = 22  # Number of possible labels
# Example shape for 8 hidden units and 8 heads
# custom_weights = torch.randn(num_features, 8 * 8)

model, metrics_tracker = initialize_and_train(
    X_train=X_train,
    y_train=y_train,
    X_val=X_test,
    y_val=y_test,
    num_features=num_features,
    num_classes=num_classes,
    batch_size=32
)

# Get the latest metrics
latest_metrics = metrics_tracker.get_latest_metrics()
print(f"Accuracy: {latest_metrics['val']['accuracy']:.4f}, Precision: {latest_metrics['val']['precision']:.4f}, Recall: {latest_metrics['val']['recall']:.4f}, F1-score: {latest_metrics['val']['f1']:.4f}, F1 Score: {latest_metrics['val']['f1']:.4f}")

# Accuracy: 0.5560, Precision: 0.4025, Recall: 0.5560, F1-score: 0.4362 (with weight adjustment)
# Accuracy: 0.5630, Precision: 0.4324, Recall: 0.5630, F1-score: 0.4412 (no weight adjustment)
