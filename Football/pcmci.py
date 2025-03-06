import tigramite.plotting as tp
import pandas as pd
import numpy as np
import pickle
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import plotting as tp
import os
from common import *
from process_data import *
import datetime
import argparse
import shutil


def runPCMCI(save_graphs=False, save_time_series_graphs=False, save_result=False):
    df = readMatchData("match_data_causal_2.csv")
    print('Data loaded')

    # all_player_results = {}
    df = df.astype(np.float64)
    print('Starting PCMCI')
    datatime = df['time'].to_numpy()
    df = df.drop(columns=['player_id', 'time'])
    var_names = df.columns

    # Set up tigramite DataFrame
    dataframe = pp.DataFrame(
        df.to_numpy(), datatime=datatime, var_names=var_names)

    # Run PCMCI
    pcmci = PCMCI(dataframe=dataframe,
                  cond_ind_test=ParCorr(significance='analytic'))
    result = pcmci.run_pcmci(tau_max=tau_max, alpha_level=0.05, pc_alpha=0.05)

    print(f'Saving results to {save_path}')

    if save_graphs:
        tp.plot_graph(
            val_matrix=result['val_matrix'],
            graph=result['graph'],
            var_names=var_names,
            link_colorbar_label='cross-MCI',
            node_colorbar_label='auto-MCI',
            show_autodependency_lags=False,
            save_name=f'{save_path}/graph.png'
        )

    if save_time_series_graphs:
        tp.plot_time_series_graph(
            figsize=(6, 4),
            val_matrix=result['val_matrix'],
            graph=result['graph'],
            var_names=var_names,
            link_colorbar_label='MCI',
            save_name=f'{save_path}/time_series_graph.png'
        )

    try:
        tp.write_csv(
            val_matrix=result['val_matrix'],
            graph=result['graph'],
            var_names=var_names,
            save_name=f'{save_path}/links.csv',
            digits=5,
        )
    except ValueError:
        print(
            f'No causality found, writing empty csv')
        with open(f'{save_path}/links.csv', 'w') as f:
            f.write(
                'Variable i,Variable j,Time lag of i,Link type i --- j,Link value')

    if save_result:
        with open(f'{save_path}/result.pkl', 'wb') as f:
            pickle.dump(result, f)

    np.save(f'{save_path}/val_matrix.npy', result['val_matrix'])
    np.save(f'{save_path}/graph.npy', result['graph'])
    print('PCMCI saved')
    return result


'''
def aggregateLinks(match_folder=None):
    print('Aggregating result')
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

    graph = createEmptyMatrix(22, 22, tau_max+1, '')
    val_matrix = createEmptyMatrix(22, 22, tau_max+1, 0.0)
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
'''


def copyFile(file_path, fname):
    shutil.copy2(file_path, f'{save_path}/{fname}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action-path', type=str, default="fkeys/action_2.csv")
    parser.add_argument('--tau', type=int, default=TAU_MAX)
    args = parser.parse_args()
    tau_max = args.tau
    action_path = args.action_path

    folder_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_path = f'saves/{folder_name}'
    os.makedirs(save_path)
    runPCMCI(save_graphs=False, save_time_series_graphs=False, save_result=False)
    copyFile(action_path, 'action_2.csv')
