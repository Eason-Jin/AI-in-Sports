from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        
        # First GCN layer
        self.conv1 = GCNConv(
            in_channels=in_channels,
            out_channels=hidden_channels
        )
        
        # Second GCN layer
        self.conv2 = GCNConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels
        )
        
        # Output layer - matching GAT's output dimension
        self.linear = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        # First GCN layer
        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index, edge_weight)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.linear(x)
        return x