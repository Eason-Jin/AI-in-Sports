
import torch
import pytorch_lightning as pl
import torch_geometric as tg


class CausalGATv2Wrapper():
    def __new__(wrapper, *args, **kwargs):
        # forbids instance creation and calls __call__ instead
        return CausalGATv2Wrapper.__call__(*args, **kwargs)

    @staticmethod
    def load_from_checkpoint(*args, num_var=None, lookback=None, **kwargs):
        return TSGNNPredictor.load_from_checkpoint(*args, num_var=num_var, lookback=lookback, gnn_class=tg.nn.GATv2Conv, **kwargs)

    @staticmethod
    def __call__(num_var, lookback, weights, hidden_size=128, masked_idxs_for_training=None):
        return TSGNNPredictor(num_var=num_var, lookback=lookback, weights=weights, hidden_size=hidden_size, gnn_class=tg.nn.GATv2Conv, masked_idxs_for_training=masked_idxs_for_training)


class TSPredictor(pl.LightningModule):
    def __init__(self, masked_idxs_for_training=None):
        super().__init__()
        self.masked_idxs_for_training = masked_idxs_for_training

    def training_step(self, batch, batch_idx):
        x, y, i = batch
        y_pred = self(x)
        if self.masked_idxs_for_training is not None:  # Remove masked variables
            y_pred = y_pred[:, :, torch.where(~torch.tensor(
                [i in self.masked_idxs_for_training for i in range(self.num_var)]))[0]]
            y_pred = y_pred.softmax(dim=-1)
            y = y[:, :, torch.where(~torch.tensor(
                [i in self.masked_idxs_for_training for i in range(self.num_var)]))[0]]

        loss = torch.nn.functional.binary_cross_entropy(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, i = batch
        y_pred = self(x)
        if self.masked_idxs_for_training is not None:  # Remove masked variables
            y_pred = y_pred[:, :, torch.where(~torch.tensor(
                [i in self.masked_idxs_for_training for i in range(self.num_var)]))[0]]
            y_pred = y_pred.softmax(dim=-1)
            y = y[:, :, torch.where(~torch.tensor(
                [i in self.masked_idxs_for_training for i in range(self.num_var)]))[0]]

        loss = torch.nn.functional.binary_cross_entropy(y_pred, y)
        self.log('val_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, i = batch
        y_pred = self(x)
        if self.masked_idxs_for_training is not None:  # Remove masked variables
            y_pred = y_pred[:, :, torch.where(~torch.tensor(
                [i in self.masked_idxs_for_training for i in range(self.num_var)]))[0]]
            y_pred = y_pred.softmax(dim=-1)

        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def backward(self, loss):
        loss.backward(retain_graph=True)


class TSGNNPredictor(TSPredictor):
    def __init__(self, num_var, lookback, weights, hidden_size=128, gnn_class=tg.nn.GCNConv, masked_idxs_for_training=None):
        super().__init__(masked_idxs_for_training)
        self.num_var = num_var
        self.lookback = lookback
        self.graph = weights  # shape is (num_var, num_var, lookback)
        self.hidden_size = hidden_size
        self.gnn_class = gnn_class

        self.input_layer = torch.nn.Linear(num_var, hidden_size)
        graph_layers = []
        for _ in range(lookback):
            graph_layers.append(gnn_class(hidden_size, hidden_size))
        self.graph_layers = torch.nn.ModuleList(graph_layers)
        self.output_layer = torch.nn.Linear(hidden_size, 1)

        self.save_hyperparameters()

    def forward(self, x):
        batch_size = x.shape[0]
        features = torch.nn.functional.one_hot(torch.arange(self.num_var)).reshape(
            (1, 1, self.num_var, self.num_var)).repeat(batch_size, self.lookback, 1, 1).to(x.device)
        features = x.unsqueeze(-1) * features
        features = self.input_layer(features)

        # shape is (batch_size, lookback, num_var, hidden_size)
        outputs = torch.zeros_like(features)
        for i, layer in enumerate(self.graph_layers):
            features_i = features[:, :i+1, :,
                                  :].view((batch_size, (i+1)*self.num_var, self.hidden_size))
            edges_i = torch.stack(torch.where(self.graph[:, :, :i+1].permute(
                2, 0, 1).reshape(((i+1)*self.num_var, self.num_var))), dim=0).to(x.device)

            # GATConv and GATv2Conv do not support static graph (see https://github.com/pyg-team/pytorch_geometric/issues/2844 and https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html)
            if isinstance(layer, tg.nn.GATConv) or isinstance(layer, tg.nn.GATv2Conv):
                data_list = [tg.data.Data(features_i[j], edges_i)
                             for j in range(len(features_i))]
                mini_batch = tg.data.Batch.from_data_list(data_list)
                batched_features_i = mini_batch.x
                batched_edges_i = mini_batch.edge_index
                outputs[:, i, :, :] = layer(batched_features_i, batched_edges_i).reshape(
                    features_i.shape)[:, :self.num_var, :]
            else:
                outputs[:, i, :, :] = layer(features_i, edges_i)[
                    :, :self.num_var, :]

        outputs = torch.nn.functional.leaky_relu(outputs)
        outputs = self.output_layer(outputs)
        outputs = torch.nn.functional.leaky_relu(outputs)

        return outputs.view((batch_size, self.lookback, self.num_var))
