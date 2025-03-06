import pandas as pd
import os
import torch

TAU_MAX = 5
CLOSE_THRESHOLD = 45
DEVICE = torch.device("cuda")
# DEVICE = torch.device("cpu")


def searchDF(df, conditions):
    """
    Finds matching rows in df similar to queries
    """
    for col, val in conditions:
        if isinstance(val, str):
            if val.startswith('!'):
                df = df[df[col] != val[1:]]
            elif val.startswith('>='):
                # Try to convert to numeric first, fall back to datetime if needed
                try:
                    numeric_val = float(val[2:])
                    df = df[df[col].astype(float) >= numeric_val]
                except ValueError:
                    df = df[pd.to_datetime(df[col], errors='coerce') >= pd.to_datetime(
                        val[2:], errors='coerce')]
            elif val.startswith('>'):
                try:
                    numeric_val = float(val[1:])
                    df = df[df[col].astype(float) > numeric_val]
                except ValueError:
                    df = df[pd.to_datetime(df[col], errors='coerce') > pd.to_datetime(
                        val[1:], errors='coerce')]
            elif val.startswith('<='):
                try:
                    numeric_val = float(val[2:])
                    df = df[df[col].astype(float) <= numeric_val]
                except ValueError:
                    df = df[pd.to_datetime(df[col], errors='coerce') <= pd.to_datetime(
                        val[2:], errors='coerce')]
            elif val.startswith('<'):
                try:
                    numeric_val = float(val[1:])
                    df = df[df[col].astype(float) < numeric_val]
                except ValueError:
                    df = df[pd.to_datetime(df[col], errors='coerce') < pd.to_datetime(
                        val[1:], errors='coerce')]
            elif val.startswith('.'):
                df = df[df[col].astype(str).str.contains(val[1:], na=False)]
            else:
                df = df[df[col] == val]
        else:
            df = df[df[col] == val]
    return df


def readMatchData(file_name, max_match_folder=10):
    """
    The equivalent of PandasFormatterEnsemble
    """
    df = pd.DataFrame()
    for folder in range(max_match_folder):
        df = pd.concat(
            [df, pd.read_csv(f'matches/match_{folder}/{file_name}')])
    return df


def get_last_subfolder(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    if not subfolders:
        raise Exception('No PCMCI saves! Run `pcmci.py` first!')
    return max(subfolders, key=os.path.getmtime)
