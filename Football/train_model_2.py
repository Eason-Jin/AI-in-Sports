import numpy as np
from common import TAU_MAX, DEVICE, get_last_subfolder, readMatchData, searchDF
from models.predictor import CausalGATv2Wrapper
import argparse
from torch.utils.data import DataLoader
from SeriesDataset import SeriesDataset
import pytorch_lightning as pl
import os
import torch
import pickle


def direct_prediction_accuracy(model, loader, num_var, masked_index):
    acc = torch.tensor(0.0)
    acc_last = torch.tensor(0.0)
    model = model.to(DEVICE)
    for i, (x, y, _) in enumerate(loader):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Make prediction
        y_pred = model(x)

        # Remove masked variables
        y_pred = y_pred[:,:,torch.where(~torch.tensor([i in masked_index for i in range(num_var)]))[0]]
        y = y[:,:,torch.where(~torch.tensor([i in masked_index for i in range(num_var)]))[0]]

        y_pred = y_pred.softmax(dim=-1)

        # Calculate accuracy
        acc = acc * i / (i+1) + (y_pred.argmax(dim=-1) ==
                                 y.argmax(dim=-1)).float().mean() / (i+1)
        acc_last = acc_last * i / (i+1) + (y_pred[:, -1, :].argmax(
            dim=-1) == y[:, -1, :].argmax(dim=-1)).float().mean() / (i+1)
    return acc, acc_last


def evaluate_model(save_path, loader, num_var, masked_index):
    with open(f"{save_path}/model.pkl", "rb") as f:
        model = pickle.load(f)
    acc, acc_last = direct_prediction_accuracy(model, loader, num_var, masked_index)
    print(f"Direct Prediction Accuracy: {torch.round(acc*100, decimals=2)}")
    print(f"Direct Prediction Accuracy (last layer only): {torch.round(acc_last*100, decimals=2)}")


def split_list(player_ids, split_ratio=0.8):
    player_ids = np.array(player_ids)
    np.random.shuffle(player_ids)

    split_index = int(len(player_ids) * split_ratio)
    train_list = player_ids[:split_index].tolist()
    test_list = player_ids[split_index:].tolist()
    return train_list, test_list


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcmci-path', type=str,
                        default=get_last_subfolder("saves"))
    parser.add_argument('--tau', type=int, default=TAU_MAX)
    parser.add_argument('--load', type=bool, default=False)
    args = parser.parse_args()
    pcmci_path = args.pcmci_path
    tau_max = args.tau
    load = args.load

    # Only considering the first match right now
    match_df = readMatchData("match_data_causal_2.csv")
    match_df.sort_values(by=['player_id', 'time'], inplace=True)
    match_df = match_df.drop(columns=['time'])

    variables = match_df.columns.to_list()
    masked_index = []
    for v in variables:
        if ("Team" in v) or ("Opponent" in v):
            masked_index.append(variables.index(v))

    train_sequences, test_sequences, val_sequences = {}, {}, {}
    
    player_ids = match_df['player_id'].unique()
    train_players, test_players = split_list(player_ids, split_ratio=0.8)
    train_players, val_players = split_list(train_players, split_ratio=0.8)
    for i in range(len(train_players)):
        train_sequences[i] = np.array(
            searchDF(match_df, [('player_id', player_ids[i])]).drop(columns=['player_id']))
    for i in range(len(test_players)):
        test_sequences[i] = np.array(
            searchDF(match_df, [('player_id', player_ids[i])]).drop(columns=['player_id']))
    for i in range(len(val_players)):
        val_sequences[i] = np.array(
            searchDF(match_df, [('player_id', player_ids[i])]).drop(columns=['player_id']))

    train_dataset = SeriesDataset(train_sequences, tau=TAU_MAX+1)
    test_dataset = SeriesDataset(test_sequences, tau=TAU_MAX+1)
    val_dataset = SeriesDataset(val_sequences, tau=TAU_MAX+1)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    if not load:
        num_var = len(match_df.columns)-1   # Minus "player_id"

        val_matrix = np.load(f'{pcmci_path}/val_matrix.npy')
        graph = np.load(f'{pcmci_path}/graph.npy')
        val_matrix = torch.nan_to_num(
            torch.from_numpy(val_matrix).float()).to(DEVICE)
        graph[np.where(graph != "-->")] = "0"
        graph[np.where(graph == "-->")] = "1"
        graph[np.where(graph == "o-o")] = "1"
        graph = graph.astype(np.int64)
        graph = torch.from_numpy(graph).float().to(DEVICE)

        weights = (graph*val_matrix).to(DEVICE)

        model = CausalGATv2Wrapper(
            num_var=num_var, tau=tau_max+1, weights=weights, masked_idxs_for_training=masked_index)
        trainer = pl.Trainer(max_epochs=10)

        trainer.fit(model, train_loader, val_loader)

        with open(f"{pcmci_path}/model.pkl", "wb") as f:
            pickle.dump(model, f)

    evaluate_model(pcmci_path, test_loader, num_var, masked_index)
