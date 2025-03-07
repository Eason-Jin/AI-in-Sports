from PCMCI.find import *
import pandas as pd
import matplotlib.pyplot as plt

searchCSV('NFL TRACKING/2017091100.csv', [('event', 'ball_snap'), ('playId', str(152))])

# Filter the data using searchCSV
play = pd.read_csv('query.csv')
if play.empty:
    print("Empty dataframe")
else:
    # Convert `jerseyNumber` to integers
    play['jerseyNumber'] = play['jerseyNumber'].astype(float).astype(int)
    play['x'] = play['x'].astype(float)
    play['y'] = play['y'].astype(float)

    plt.figure()

    # Determine axis limits
    x_min, x_max = play['x'].min(), play['x'].max()
    y_min, y_max = play['y'].min(), play['y'].max()

    # Plot each row in the filtered data
    for _, row in play.iterrows():
        color = 'blue' if row['team'] == 'home' else 'yellow' if row['team'] == 'away' else 'grey'
        plt.scatter(row['x'], row['y'], color=color)
        plt.text(row['x'], row['y'], str(row['jerseyNumber']), 
                fontsize=12, ha='center', va='center', color='black', weight='bold')

    # Configure the plot
    plt.xlim(x_min - 1, x_max + 1)  # Add padding around the x-axis
    plt.ylim(y_min - 1, y_max + 1)  # Add padding around the y-axis
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Players Plot')
    plt.grid(True)
    plt.show()
