import os
import pandas as pd
import numpy as np
from common import *


def matchData():
    # Splits the match data into individual matches and saves them in separate folders
    df = pd.read_csv('match_event.csv')
    # Only consider events that are successful
    df = searchDF(df, [('is_success', 't')])
    df['eventsec'] = df['eventsec'].apply(lambda x: round(x))
    df.fillna({'eventname': 'None'}, inplace=True)
    df.fillna({'action': 'None'}, inplace=True)
    df.fillna({'modifier': 'None'}, inplace=True)
    df = df.drop(columns=['is_success'])

    match_period = df['matchperiod'].unique()
    match_map = {'1H': 1,
                 '2H': 2}

    event_df = pd.DataFrame(columns=['id', 'event'])
    events = df['eventname'].unique()
    events = np.sort(events)
    event_map = {events[i]: i for i in range(len(events))}
    event_df['eventname'] = events
    event_df['id'] = event_map.values()
    event_df.to_csv('fkeys/event.csv', index=False)

    action_df = pd.DataFrame(columns=['id', 'action'])
    actions = df['action'].unique()
    actions = np.sort(actions)
    action_map = {actions[i]: i for i in range(len(actions))}
    action_df['action'] = actions
    action_df['id'] = action_map.values()
    action_df.to_csv('fkeys/action.csv', index=False)

    modifier_df = pd.DataFrame(columns=['id', 'modifier'])
    modifiers = df['modifier'].unique()
    modifiers = np.sort(modifiers)
    modifier_map = {modifiers[i]: i for i in range(len(modifiers))}
    modifier_df['modifier'] = modifiers
    modifier_df['id'] = modifier_map.values()
    modifier_df.to_csv('fkeys/modifier.csv', index=False)

    df['matchperiod'] = df['matchperiod'].apply(lambda x: match_map[x])
    df['eventname'] = df['eventname'].apply(lambda x: event_map[x])
    df['action'] = df['action'].apply(lambda x: action_map[x])
    df['modifier'] = df['modifier'].apply(lambda x: modifier_map[x])

    df.rename(columns={'eventname': 'event'}, inplace=True)
    df.rename(columns={'matchperiod': 'match_period'}, inplace=True)
    df.rename(columns={'eventsec': 'time'}, inplace=True)
    df.rename(columns={'modifier': 'action_result'}, inplace=True)
    df.rename(columns={'players_id': 'player_id'}, inplace=True)

    df = df.sort_values(by=['id'])
    df.to_csv('success.csv', index=False)

    match_ids = df['match_id'].unique()
    for match_id in match_ids:
        match_df = searchDF(df, [('match_id', match_id)])
        first_half = match_df[match_df['match_period'] == 1]
        max_time = first_half['time'].max()
        match_df.loc[match_df['match_period'] == 2, 'time'] += max_time
        match_df = match_df.drop(
            columns=['match_id', 'match_period', 'club_id'])
        min_time = match_df['time'].min()
        match_df['time'] -= min_time
        # convert to minutes
        match_df['time'] = (match_df['time'] / 60).round().astype(int)
        match_df = match_df.sort_values(by=['player_id', 'time'])
        path = f'matches/match_{match_id}'
        if not os.path.exists(path):
            os.makedirs(path)
        match_df.to_csv(
            f'matches/match_{match_id}/match_data.csv', index=False)


def causalData(match_folder):
    df = pd.read_csv(f'matches/{match_folder}/match_data.csv')
    df = df.drop(columns=['id', 'event',
                          'action_result', 'x_begin', 'y_begin', 'x_end', 'y_end'])

    action_df = pd.read_csv('fkeys/action.csv')
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
            temp_df.loc[temp_df['time'] == time,
                        action_df['action'][action]] = 1
        new_df = pd.concat([new_df, temp_df], ignore_index=True)
    new_df.to_csv(f'matches/{match_folder}/match_data_causal.csv', index=False)
    return new_df


def mlpData(match_folder):
    # time, x_begin, y_begin, x_end, y_end, prev_action, time_lag, action(prediction)
    df = pd.read_csv(f'matches/{match_folder}/match_data.csv')
    df = df.drop(columns=['id', 'event', 'action_result'])
    action_column = df['action']
    df = df.drop(columns=['action'])
    # Find the previous action that this player took and the time of that action
    prev_action = action_column.shift(1)
    prev_time = df['time'].shift(1)
    # Find the time difference between the current action and the previous action
    time_lag = df['time'] - prev_time
    df['prev_action'] = prev_action
    df['time_lag'] = time_lag
    # Drop the first row of each player
    players = df['player_id'].unique()
    for player in players:
        indices = df[df['player_id'] == player].index
        df = df.drop(indices[0])
    df = df.drop(columns=['time', 'player_id'])
    df['action'] = action_column
    df.to_csv(f'matches/{match_folder}/match_data_mlp.csv', index=False)
    return df


def gatData(match_folder):
    if not os.path.exists(f'matches/{match_folder}/match_data_causal.csv'):
        causalData(match_folder)
    df = pd.read_csv(f'matches/{match_folder}/match_data_causal.csv')
    columns = [f'action_(t-{TAU_MAX-i})' for i in range(TAU_MAX)]
    columns.extend(['result'])
    results = []
    # read TAU_MAX+1 rows at a time
    for index in range(TAU_MAX, len(df)):
        temp_df = df.iloc[index-TAU_MAX:index+1]
        actions = []
        skip = False
        for i in range(len(temp_df)):
            action = list(np.where(temp_df.iloc[i, 2:] == 1)[0])
            if len(action) == 0 and i == 0:
                skip = True
                break
            # Interpolate missing actions
            actions.append(action if len(action) > 0 else actions[i-1])
        if not skip:
            results = expand_2d_list(results, actions)
    gat_df = pd.DataFrame(results, columns=columns)
    gat_df.to_csv(f'matches/{match_folder}/match_data_gat.csv', index=False)


def expand_2d_list(results, lst):
    # Base case: if every element in lst is length 1, return lst as 1 list
    if all([len(elem) == 1 for elem in lst]):
        results.append([elem[0] for elem in lst])
    # Recursive case:
    else:
        # Find the first element in lst that is not length 1
        for i, elem in enumerate(lst):
            if len(elem) != 1:
                # For each element in elem, create a new list with that element
                for item in elem:
                    new_lst = lst.copy()
                    new_lst[i] = [item]
                    expand_2d_list(results, new_lst)
                break
    return results


if __name__ == '__main__':
    # matchData()
    max_match_folder = 10
    for m in range(max_match_folder):
        print(f'Processing match_{m}')
        match_folder = f'match_{m}'
        causalData(match_folder)
        # mlpData(match_folder)
        gatData(match_folder)
