from enum import Enum
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

TAU_MAX = 10


def apply_conditions(df, conditions):
    for col, val in conditions:
        if isinstance(val, str):
            if val.startswith('!'):
                df = df[df[col] != val[1:]]
            elif val.startswith('>='):
                df = df[pd.to_datetime(df[col], errors='coerce') >= pd.to_datetime(
                    val[2:], errors='coerce')]
            elif val.startswith('>'):
                df = df[pd.to_datetime(df[col], errors='coerce') > pd.to_datetime(
                    val[1:], errors='coerce')]
            elif val.startswith('<='):
                df = df[pd.to_datetime(df[col], errors='coerce') <= pd.to_datetime(
                    val[2:], errors='coerce')]
            elif val.startswith('<'):
                df = df[pd.to_datetime(df[col], errors='coerce') < pd.to_datetime(
                    val[1:], errors='coerce')]
            elif val.startswith('.'):
                df = df[df[col].astype(str).str.contains(val[1:], na=False)]
            else:
                df = df[df[col] == val]
        else:
            df = df[df[col] == val]
    return df


def searchCSV(source_fname, conditions, result_fname='query.csv'):
    try:
        df = pd.read_csv(source_fname)
        result_df = apply_conditions(df, conditions)
        result_df.to_csv(result_fname, index=False)
        return result_df
    except Exception as e:
        print(e)


def searchDF(source_df, conditions):
    return apply_conditions(source_df, conditions)


def aggregate_links(group):
    # Define priority hierarchy for link types
    priority_order = [
        # Highest priority: Directed links
        '-->', '<--', '<->',
        # Medium priority: Partially directed links
        'o->', '<-o', 'x->', '<-x', '<-+', '+->',
        # Lower priority: Confounded links
        'x-o', 'o-x', 'x--', '--x', 'x-x',
        # Lowest priority: Undirected links
        'o-o', 'o--', '--o', '---'
    ]

    # Find the highest priority link type in the group
    for link_type in priority_order:
        if any(group['Link type i --- j'] == link_type):
            selected_link_type = link_type
            break
    else:
        # Default to '---' if no link type is found (should not happen)
        selected_link_type = '---'

    # Average the strengths
    avg_strength = round(group['Link value'].mean(), 5)

    return pd.Series({
        'Link type i --- j': selected_link_type,
        'Link value': avg_strength
    })


def createEmptyMatrix(width, height, depth, initValue):
    return np.full((width, height, depth), initValue)


def printMetrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(
        f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    return accuracy, precision, recall, f1


def readMatchData(file_name, max_match_folder=10):
    df = pd.DataFrame()
    for folder in range(max_match_folder):
        df = pd.concat(
            [df, pd.read_csv(f'matches/match_{folder}/{file_name}')])
    return df


class ModelTypes(Enum):
    GATv2 = 'gat'
    GCN = 'gcn'


class LinkTypes(Enum):
    RANDOM = 'random'
    PCMCI = 'pcmci'
    INVERSE_COUNT = 'inverse_count'


def get_enum_key(enum, value):
    for key, member in enum.__members__.items():
        if member.value == value:
            return key
    return None


def get_last_subfolder(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    if not subfolders:
        raise Exception('No PCMCI saves! Run `pcmci.py` first!')
    return max(subfolders, key=os.path.getmtime)
