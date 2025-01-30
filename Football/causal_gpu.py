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

print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device != torch.device('cuda'):
    raise ValueError("CUDA not available!")

TAU_MAX = 5
match_folder = 'match_0'
df = pd.read_csv(f'Football/matches/{match_folder}/match_data.csv')

df = df.sort_values(by=['player_id', 'time'])

# Construct new df where each row represents a player's actions in a time window

dataframe = pp.DataFrame(
    time_series_data, var_names=var_names, analysis_mode='multiple')
pcmci = PCMCI(dataframe=dataframe,
              cond_ind_test=ParCorr(significance='analytic'))
print('Starting PCMCI')
results = pcmci.run_pcmci(tau_max=TAU_MAX)
q_matrix = pcmci.get_corrected_pvalues(
    p_matrix=results['p_matrix'], tau_max=8, fdr_method='fdr_bh')
pcmci.print_significant_links(
    p_matrix=q_matrix,
    val_matrix=results['val_matrix'],
    alpha_level=0.01)
graph = pcmci.get_graph_from_pmatrix(p_matrix=q_matrix, alpha_level=0.01,
                                     tau_min=0, tau_max=TAU_MAX, link_assumptions=None)
results['graph'] = graph
with open(f'Football/matches/{match_folder}/results.pkl', 'wb') as f:
    pickle.dump(results, f)
print('PCMCI saved')

print('Plotting graph')
tp.plot_graph(
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    show_autodependency_lags=False,
    save_name=f'Football/matches/{match_folder}/graph.png'
)

print('Plotting time series graph')
tp.plot_time_series_graph(
    figsize=(6, 4),
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    var_names=var_names,
    link_colorbar_label='MCI',
    save_name=f'Football/matches/{match_folder}/time_series_graph.png'
)
