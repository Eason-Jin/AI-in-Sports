import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
import math

class GATv2Layer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout=0.6, alpha=0.2):
        super(GATv2Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha

        # Linear transformations
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features * num_heads)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def set_input_weights(self, custom_weights):
        assert custom_weights.shape == self.W.shape, f"Weight matrix shape mismatch. Expected {self.W.shape}, got {custom_weights.shape}"
        self.W.data = torch.tensor(custom_weights, dtype=torch.float32)

    def forward(self, x, batch_size):
        # x shape: (batch_size, in_features)
        
        # Linear transformation
        h = torch.mm(x, self.W)
        h = h.view(batch_size, self.num_heads, -1)
        
        # Create attention for batch
        # More memory-efficient way to compute self-attention
        q = h
        k = h
        
        # Compute attention scores
        attention = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.out_features)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)
        
        # Apply attention
        out = torch.bmm(attention, h)
        return out.mean(dim=1)  # Average over heads

class GATv2(nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=8, num_heads=8, dropout=0.6):
        super(GATv2, self).__init__()
        self.dropout = dropout
        
        # First GAT layer
        self.gat1 = GATv2Layer(num_features, num_hidden, num_heads, dropout=dropout)
        
        # Output layer
        self.out = nn.Linear(num_hidden, num_classes)
        
    def set_input_weights(self, custom_weights):
        self.gat1.set_input_weights(custom_weights)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, batch_size))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x)
        return F.log_softmax(x, dim=1)

def evaluate_metrics(model, data_loader):
    """Calculate performance metrics for the model using batched data"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            output = model(features)
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_preds, 
        average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

class MetricsTracker:
    def __init__(self):
        self.train_metrics = []
        self.val_metrics = []
    
    def update(self, train_metrics, val_metrics=None):
        self.train_metrics.append(train_metrics)
        if val_metrics:
            self.val_metrics.append(val_metrics)
    
    def get_latest_metrics(self):
        return {
            'train': self.train_metrics[-1] if self.train_metrics else None,
            'val': self.val_metrics[-1] if self.val_metrics else None
        }

def train_gatv2(model, optimizer, train_loader, val_loader=None, epochs=200):
    """Train the GATv2 model using batched data"""
    metrics_tracker = MetricsTracker()
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            output = model(features)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Calculate metrics
        train_metrics = evaluate_metrics(model, train_loader)
        train_metrics['loss'] = total_loss / len(train_loader)
        
        # Validation metrics
        val_metrics = None
        if val_loader:
            val_metrics = evaluate_metrics(model, val_loader)
        
        # Update metrics tracker
        metrics_tracker.update(train_metrics, val_metrics)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Training - Loss: {train_metrics["loss"]:.4f}, '
                  f'Accuracy: {train_metrics["accuracy"]:.4f}, '
                  f'F1: {train_metrics["f1"]:.4f}')
            if val_metrics:
                print(f'Validation - Accuracy: {val_metrics["accuracy"]:.4f}, '
                      f'F1: {val_metrics["f1"]:.4f}')
            print('-' * 50)
    
    return metrics_tracker

def prepare_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """Prepare DataLoaders for training and validation"""
    # Convert to PyTorch tensors
    train_features = torch.FloatTensor(X_train.values)
    train_labels = torch.LongTensor(y_train.values)
    
    # Create datasets
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_features = torch.FloatTensor(X_val.values)
        val_labels = torch.LongTensor(y_val.values)
        val_dataset = TensorDataset(val_features, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def initialize_and_train(X_train, y_train, X_val, y_val, num_features, num_classes, 
                        batch_size=32, custom_weights=None):
    """Initialize and train the model with batched data"""
    # Prepare data loaders
    train_loader, val_loader = prepare_data_loaders(
        X_train, y_train, X_val, y_val, batch_size
    )
    
    # Initialize model
    model = GATv2(num_features=num_features,
                  num_classes=num_classes,
                  num_hidden=8,
                  num_heads=8,
                  dropout=0.6)
    
    # Set custom weights if provided
    if custom_weights is not None:
        model.set_input_weights(custom_weights)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    
    # Train the model
    metrics_tracker = train_gatv2(
        model, 
        optimizer, 
        train_loader, 
        val_loader
    )
    
    return model, metrics_tracker