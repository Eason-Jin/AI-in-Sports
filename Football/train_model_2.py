import numpy as np
from common import *
from models.predictor import *
import argparse
from torch.utils.data import DataLoader
from SeriesDataset import SeriesDataset
import pytorch_lightning as pl
import os

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcmci-path', type=str,
                        default=get_last_subfolder("saves"))
    parser.add_argument('--tau', type=int, default=TAU_MAX)
    args = parser.parse_args()
    pcmci_path = args.pcmci_path
    model_type = args.model_type
    tau_max = args.tau

    # Only considering the first match right now
    match_df = readMatchData("match_data_causal_2.csv", max_match_folder=1)
    match_df.sort_values(by=['player_id', 'time'], inplace=True)
    match_df = match_df.drop(columns=['time'])

    num_var = len(match_df.columns)-1

    val_matrix = np.load(f'{pcmci_path}/val_matrix.npy')
    graph = np.load(f'{pcmci_path}/graph.npy')
    val_matrix = torch.nan_to_num(torch.from_numpy(val_matrix).float()).to(DEVICE)
    graph[np.where(graph != "-->")] = "0"
    graph[np.where(graph == "-->")] = "1"
    graph = graph.astype(np.int64)
    graph = torch.from_numpy(graph).float().to(DEVICE)

    weights = (graph*val_matrix).to(DEVICE)

    train_sequences = {}
    player_ids = match_df['player_id'].unique()
    for i in range(len(player_ids)):
        train_sequences[i] = np.array(
            searchDF(match_df, [('player_id', player_ids[i])]).drop(columns=['player_id']))
    train_dataset = SeriesDataset(train_sequences, tau=TAU_MAX+1)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = CausalGATv2Wrapper(
        num_var=num_var, tau=tau_max+1, weights=weights).to(DEVICE)
    trainer = pl.Trainer(max_epochs=10)

    trainer.fit(model, train_loader)
