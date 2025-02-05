import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dense_to_sparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os


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

# Convert to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)


# Define the GAT model


class GATLayer(MessagePassing):
    def __init__(self, in_features, out_features, num_heads):
        super(GATLayer, self).__init__(aggr='add')  # Aggregation method
        self.num_heads = num_heads
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        # Attention parameters
        self.a = nn.Parameter(torch.randn(num_heads, 2 * out_features))

    def forward(self, x, edge_index):
        # Apply linear transformation
        x = self.W(x).view(-1, self.num_heads, x.shape[1])
        row, col = edge_index
        edge_features = torch.cat(
            [x[row], x[col]], dim=-1)  # Concatenate features

        # Compute attention
        attention_scores = (edge_features * self.a).sum(dim=-1)
        attention_scores = F.softmax(
            attention_scores, edge_index[0])  # Normalize

        return self.propagate(edge_index, x=x, attention_scores=attention_scores)

    def message(self, x_j, attention_scores):
        # Weighted sum of neighbours
        return attention_scores.unsqueeze(-1) * x_j


class GAT(nn.Module):
    def __init__(self, num_features, num_heads, num_classes):
        super(GAT, self).__init__()
        self.gat1 = GATLayer(num_features, num_features, num_heads)
        self.gat2 = GATLayer(num_features, num_classes,
                             1)  # Single-head final layer

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x


# Initialize the model
num_heads = 4  # Number of attention heads
num_features = X_train.shape[1]
num_classes = 22  # Number of possible labels
model = GAT(num_features=num_features,
            num_heads=num_heads, num_classes=num_classes)

# Define a simple training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

num_nodes = X_train.shape[0]  # Number of data points
# Fully connected (except self-loops)
adjacency_matrix = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
edge_index = dense_to_sparse(adjacency_matrix)[0]
# Training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(X_train, edge_index)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test, edge_index)
    y_pred = y_pred.argmax(dim=1).numpy()

# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Accuracy: 0.5560, Precision: 0.4025, Recall: 0.5560, F1-score: 0.4362 (with weight adjustment)
# Accuracy: 0.5630, Precision: 0.4324, Recall: 0.5630, F1-score: 0.4412 (no weight adjustment)
