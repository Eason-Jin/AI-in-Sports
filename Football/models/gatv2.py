from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
import torch.nn as nn

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