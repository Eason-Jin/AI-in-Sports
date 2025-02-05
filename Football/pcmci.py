from tigramite.models import Prediction
import sklearn
import tigramite.plotting as tp
import pandas as pd
import numpy as np
import torch
import pickle
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import plotting as tp
import os
import sys
from common import *
from process_data import *
import matplotlib.pyplot as plt


def runPCMCI(match_folder, save_graphs=False, save_time_series_graphs=False, save_results=False):
    df = pd.read_csv(f'matches/{match_folder}/match_data_causal.csv')

    if not os.path.exists(f'matches/{match_folder}/graphs') and save_graphs:
        os.makedirs(f'matches/{match_folder}/graphs')
    if not os.path.exists(f'matches/{match_folder}/links'):
        os.makedirs(f'matches/{match_folder}/links')
    if not os.path.exists(f'matches/{match_folder}/time_series_graphs') and save_time_series_graphs:
        os.makedirs(f'matches/{match_folder}/time_series_graphs')

    all_player_results = {}

    print('Starting PCMCI')
    players = df['player_id'].unique()
    for player_id in players:
        print(f'Player {player_id}')
        # Extract data for the current player
        player_df = searchDF(df, [('player_id', player_id)]).astype(np.float64)
        datatime = player_df['time'].to_numpy()
        player_df = player_df.drop(columns=['player_id', 'time'])
        var_names = player_df.columns

        # Set up tigramite DataFrame
        dataframe = pp.DataFrame(
            player_df.to_numpy(), datatime=datatime, var_names=var_names)

        # Run PCMCI
        pcmci = PCMCI(dataframe=dataframe,
                      cond_ind_test=ParCorr(significance='analytic'))
        results = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=None)
        q_matrix = pcmci.get_corrected_pvalues(
            p_matrix=results['p_matrix'], tau_max=TAU_MAX, fdr_method='fdr_bh')

        graph = pcmci.get_graph_from_pmatrix(p_matrix=q_matrix, alpha_level=0.01,
                                             tau_min=0, tau_max=TAU_MAX)
        results['graph'] = graph

        all_player_results[player_id] = results

        if save_graphs:
            tp.plot_graph(
                val_matrix=results['val_matrix'],
                graph=results['graph'],
                var_names=var_names,
                link_colorbar_label='cross-MCI',
                node_colorbar_label='auto-MCI',
                show_autodependency_lags=False,
                save_name=f'matches/{match_folder}/graphs/graph_{player_id}.png'
            )

        if save_time_series_graphs:
            tp.plot_time_series_graph(
                figsize=(6, 4),
                val_matrix=results['val_matrix'],
                graph=results['graph'],
                var_names=var_names,
                link_colorbar_label='MCI',
                save_name=f'matches/{match_folder}/time_series_graphs/time_series_graph_{player_id}.png'
            )

        try:
            tp.write_csv(
                val_matrix=results['val_matrix'],
                graph=results['graph'],
                var_names=var_names,
                save_name=f'matches/{match_folder}/links/link_{player_id}.csv',
                digits=5,
            )
        except ValueError:
            print(
                f'No causality found for player {player_id}, writing empty csv')
            with open(f'matches/{match_folder}/links/link_{player_id}.csv', 'w') as f:
                f.write(
                    'Variable i,Variable j,Time lag of i,Link type i --- j,Link value')

    if save_results:
        with open(f'matches/{match_folder}/results.pkl', 'wb') as f:
            pickle.dump(all_player_results, f)
    print('PCMCI saved')
    return all_player_results


def aggregateLinks(match_folder=None):
    print('Aggregating results')
    if match_folder is not None:
        links_folder = f'matches/{match_folder}/links'
        link_files = [os.path.join(links_folder, file)
                      for file in os.listdir(links_folder)]
        link_dfs = [pd.read_csv(file) for file in link_files]

        combined_df = pd.concat(link_dfs, ignore_index=True)
    else:
        # read all subfolder under matches, if subfolder contains links folder, read all csv files in links folder and combine them
        link_folders = [f.path for f in os.scandir('matches') if f.is_dir()]
        link_files = []
        for folder in link_folders:
            if os.path.exists(f'{folder}/links'):
                link_files.extend([os.path.join(folder, 'links', file)
                                   for file in os.listdir(f'{folder}/links')])
        link_dfs = [pd.read_csv(file) for file in link_files]
        combined_df = pd.concat(link_dfs, ignore_index=True)

    aggregated_df = combined_df.groupby(
        ['Variable i', 'Variable j', 'Time lag of i']).apply(aggregate_links).reset_index()
    aggregated_df.to_csv(
        f'aggregated_links.csv', index=False)
    print('Aggregated results saved')

    print('Reconstructing graph')

    graph = createEmptyMatrix(22, 22, TAU_MAX+1, '')
    val_matrix = createEmptyMatrix(22, 22, TAU_MAX+1, 0.0)
    actions = (pd.read_csv('fkeys/action.csv')['action']).tolist()
    links = pd.read_csv(f'aggregated_links.csv')
    for index, row in links.iterrows():
        i = row['Variable i']
        j = row['Variable j']
        tau = int(row['Time lag of i'])
        link_type = row['Link type i --- j']
        link_value = row['Link value']
        graph[actions.index(i)][actions.index(j)][tau] = link_type
        val_matrix[actions.index(i)][actions.index(j)][tau] = link_value
        val_matrix[actions.index(j)][actions.index(i)][tau] = link_value

    tp.plot_graph(
        val_matrix=val_matrix,
        graph=graph,
        var_names=actions,
        link_colorbar_label='cross-MCI',
        node_colorbar_label='auto-MCI',
        show_autodependency_lags=False,
        save_name=f'graph.png'
    )

    tp.plot_time_series_graph(
        figsize=(6, 4),
        val_matrix=val_matrix,
        graph=graph,
        var_names=actions,
        link_colorbar_label='MCI',
        save_name=f'time_series_graph.png'
    )
    print('Graph saved')
    return graph, val_matrix

if __name__ == '__main__':
    match_folder = 'match_0'
    runPCMCI(match_folder)
    aggregateLinks(match_folder)

# Prediction
# temp_df = df.drop(columns=['player_id', 'time'])
# var_names = temp_df.columns
# full_dataframe = pp.DataFrame(
#     temp_df.astype(np.float64).to_numpy(), datatime=df['time'].to_numpy(), var_names=var_names)
# T = list(full_dataframe.T.values())[0]


# pred = Prediction(dataframe=full_dataframe,
#                   cond_ind_test=ParCorr(),
#                   prediction_model=sklearn.linear_model.LinearRegression(),
#                   data_transform=sklearn.preprocessing.StandardScaler(),
#                   train_indices=range(int(0.8*T)),
#                   test_indices=range(int(0.9*T), T),
#                   verbosity=1
#                   )

# predictors = pred.get_predictors(
#     tau_max=TAU_MAX,
#     pc_alpha=None
# )

# pred.fit(target_predictors=predictors,
#          tau_max=TAU_MAX)

# target = [i for i in range(len(var_names))]
# predicted = pred.predict(target)
# true_data = []
# for j in target:
#     true_data.append(pred.get_test_array(j=j)[0])
# # print(predicted.shape)
# # print(np.array(true_data).shape)
# true_data = np.array(true_data).T
# plt.scatter(true_data, predicted)
# plt.title(r"NMAE = %.2f" %
#           (np.abs(true_data - predicted).mean()/true_data.std()))
# plt.plot(true_data, true_data, 'k-')
# plt.xlabel('True test data')
# plt.ylabel('Predicted test data')
# plt.show()
# mae = np.mean(np.abs(true_data - predicted))
# mse = np.mean((true_data - predicted) ** 2)
# rmse = np.sqrt(mse)
# ss_res = np.sum((true_data - predicted) ** 2)
# ss_tot = np.sum((true_data - np.mean(true_data)) ** 2)
# r2 = 1 - (ss_res / ss_tot)
# nmae = mae / np.std(true_data)
# threshold = 0.05  # 5% threshold
# accuracy = np.mean(np.abs((true_data - predicted) / true_data) < threshold)
# print(accuracy)
