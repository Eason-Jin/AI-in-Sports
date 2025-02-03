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


# Process data
df = pd.read_csv(f'matches/{match_folder}/match_data.csv')
df = df.drop(columns=['id', 'event',
             'action_result', 'x_begin', 'y_begin', 'x_end', 'y_end'])

action_df = pd.read_csv('fkeys/action.csv')
action_map = {action_df['action'][i]: action_df['id'][i]
              for i in range(len(action_df))}
columns = ['player_id', 'time']
columns.extend(action_df['action'].values)
new_df = pd.DataFrame(columns=columns)
min_time = df['time'].min()
max_time = df['time'].max()

player_ids = df['player_id'].unique()
for player_id in player_ids:
    temp_df = pd.DataFrame(columns=columns)
    temp_df['player_id'] = [player_id]*((max_time-min_time)+1)
    temp_df['time'] = np.arange(min_time, max_time+1)
    temp_df = temp_df.fillna(0)
    # perform one hot encoding of action at respective times
    player_df = searchDF(df, [('player_id', player_id)])
    for index, row in player_df.iterrows():
        action = row['action']
        time = row['time']
        temp_df.loc[temp_df['time'] == time, action_df['action'][action]] = 1
    new_df = pd.concat([new_df, temp_df], ignore_index=True)
new_df.to_csv('matches/match_0/match_data_causal.csv', index=False)


df = pd.read_csv(f'Football/matches/{match_folder}/match_data_causal.csv')

if not os.path.exists(f'Football/matches/{match_folder}/graphs'):
    os.makedirs(f'Football/matches/{match_folder}/graphs')
if not os.path.exists(f'Football/matches/{match_folder}/links'):
    os.makedirs(f'Football/matches/{match_folder}/links')
if not os.path.exists(f'Football/matches/{match_folder}/time_series_graphs'):
    os.makedirs(f'Football/matches/{match_folder}/time_series_graphs')

all_player_results = {}

print('Starting PCMCI')
players = df['player_id'].unique()
# players =[24]
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

    tp.plot_graph(
        val_matrix=results['val_matrix'],
        graph=results['graph'],
        var_names=var_names,
        link_colorbar_label='cross-MCI',
        node_colorbar_label='auto-MCI',
        show_autodependency_lags=False,
        save_name=f'Football/matches/{match_folder}/graphs/graph_{player_id}.png'
    )

    tp.plot_time_series_graph(
        figsize=(6, 4),
        val_matrix=results['val_matrix'],
        graph=results['graph'],
        var_names=var_names,
        link_colorbar_label='MCI',
        save_name=f'Football/matches/{match_folder}/time_series_graphs/time_series_graph_{player_id}.png'
    )
    try:
        tp.write_csv(
            val_matrix=results['val_matrix'],
            graph=results['graph'],
            var_names=var_names,
            save_name=f'Football/matches/{match_folder}/links/link_{player_id}.csv',
            digits=5,
        )
    except ValueError:
        print(f'No causality found for player {player_id}, writing empty csv')
        with open(f'Football/matches/{match_folder}/links/link_{player_id}.csv', 'w') as f:
            f.write(
                'Variable i,Variable j,Time lag of i,Link type i --- j,Link value')

with open(f'Football/matches/{match_folder}/results.pkl', 'wb') as f:
    pickle.dump(all_player_results, f)
print('PCMCI saved')

print('Aggregating results')
links_folder = f'matches/{match_folder}/links'
link_files = [os.path.join(links_folder, file)
              for file in os.listdir(links_folder)]
link_dfs = [pd.read_csv(file) for file in link_files]

combined_df = pd.concat(link_dfs, ignore_index=True)


aggregated_df = combined_df.groupby(
    ['Variable i', 'Variable j', 'Time lag of i']).apply(aggregate_links).reset_index()
aggregated_df.to_csv(
    f'matches/{match_folder}/aggregated_links.csv', index=False)
print('Aggregated results saved')

print('Reconstructing graph')


graph = createEmptyMatrix(22, 22, TAU_MAX+1, '')
val_matrix = createEmptyMatrix(22, 22, TAU_MAX+1, 0.0)
actions = (pd.read_csv('fkeys/action.csv')['action']).tolist()
links = pd.read_csv('matches/match_0/aggregated_links.csv')
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
    save_name=f'matches/{match_folder}/graph.png'
)

tp.plot_time_series_graph(
    figsize=(6, 4),
    val_matrix=val_matrix,
    graph=graph,
    var_names=actions,
    link_colorbar_label='MCI',
    save_name=f'matches/{match_folder}/time_series_graph.png'
)
print('Graph saved')
