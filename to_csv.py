import os
import pandas as pd

folder_path = "NFL TRACKING"

for file_name in os.listdir(folder_path):
    if (not file_name.endswith('.csv')) and file_name.isnumeric:
        file_path = os.path.join(folder_path, file_name)
        
        df = pd.read_feather(file_path)
        
        # df = df[['time', 'nflId', 'gameId']]
        # print(df.head(1))
        # df.drop(columns=['time', 'nflId', 'gameId'], inplace=True)
        df = df[df['displayName'] != 'football']
        
        new_names = {
            's': 'speed',
            'dis': 'distance',
            'dir': 'direction',
            'frame.id': 'frameId'
        }
        df.rename(columns=new_names, inplace=True)
        
        csv_file_path = f"{file_path}.csv"
        df.to_csv(csv_file_path, index=True)