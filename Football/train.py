from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from common import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# print(torch.cuda.get_device_name(0))
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if device != torch.device('cuda'):
#     raise ValueError("CUDA not available!")

# Load dataset
df = pd.read_csv(f'matches/{match_folder}/match_data_gat.csv')

# Load PCMCI link file
link_file = pd.read_csv(f"matches/{match_folder}/aggregated_links.csv")

# Extract necessary columns
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values  # Target action

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Construct edge indices and edge attributes (time_lag)
edge_index = []  # List of edges (source, target)
edge_attr = []   # List of edge attributes (time_lag)

for _, row in df.iterrows():
    source = int(row['prev_action'])  # Variable i
    target = int(row['action'])       # Variable j
    time_lag = int(row['time_lag'])   # Time lag
    edge_index.append([source, target])
    edge_attr.append(time_lag)

# Convert to tensors
edge_index = torch.tensor(
    edge_index, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

# Create a Data object
data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y)

num_nodes = X.shape[0]
weight_matrix = torch.zeros(data.edge_index.shape[1], dtype=torch.float32)

# Fill the weight matrix based on the link file
action_df = pd.read_csv('fkeys/action.csv')
action_map = {action_df['action'][i]: action_df['id'][i]
              for i in range(len(action_df))}
for _, row in link_file.iterrows():
    i = action_map[row['Variable i']]
    j = action_map[row['Variable j']]
    time_lag = int(row['Time lag of i'])
    link_value = float(row['Link value'])

    # Find edges with the same time_lag and set weights
    mask = (data.edge_attr == time_lag).cpu().numpy()
    edges_with_time_lag = data.edge_index[:, mask].cpu().numpy()

    for edge in edges_with_time_lag.T:
        if edge[0] == i and edge[1] == j:
            weight_matrix[mask] = link_value

data.edge_attr = weight_matrix


class GATv2Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GATv2Model, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=1)  # edge_dim for edge_attr
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Define the model
model = GATv2Model(in_channels=X.shape[1], hidden_channels=8, out_channels=len(action_map))


# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training loop


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()


# Train the model
for epoch in range(200):
    loss = train()
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

# Metrics
model.eval()
out = model(data.x, data.edge_index, data.edge_attr)
y_pred = out.argmax(dim=1)
y_true = data.y
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
# Accuracy: 0.5560, Precision: 0.4025, Recall: 0.5560, F1-score: 0.4362