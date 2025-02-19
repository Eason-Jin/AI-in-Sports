
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from common import *
from process_data import *
from sklearn.model_selection import train_test_split


def runMLP(layers, epochs, batch_size, weight_adjustment=True):
    match_folders = [f for f in os.listdir(
        'matches') if os.path.isdir(os.path.join('matches', f))]
    df_list = []
    for match_folder in match_folders:
        if os.path.exists(f'matches/{match_folder}/match_data_mlp.csv'):
            print(f'Processing {match_folder}')
            df_list.append(pd.read_csv(
                f'matches/{match_folder}/match_data_mlp.csv'))
    df = pd.concat(df_list, ignore_index=True)

    action_df = pd.read_csv('fkeys/action.csv')
    action_map = {action_df['action'][i]: action_df['id'][i]
                  for i in range(len(action_df))}

    # Extract necessary columns
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]  # Target action

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(
        shape=(X.shape[1],)))
    for units, activation in layers:
        model.add(tf.keras.layers.Dense(units, activation=activation))
    model.add(tf.keras.layers.Dense(len(action_map)))

    model.build(input_shape=(None, X.shape[1]))

    # Initialise weights for the first layer (prev_action → action mapping)
    # Get default weights
    first_layer_weights = model.layers[0].get_weights()
    weight_matrix, bias = first_layer_weights

    weights_dict = {}
    link_file = pd.read_csv(f"links.csv")

    for _, row in link_file.iterrows():
        i, j, lag, link_value = row["Variable i"], row["Variable j"], row["Time lag of i"], row["Link value"]
        if weight_adjustment:
            weights_dict[(i, j, lag)] = link_value
        else:
            weights_dict[(i, j, lag)] = np.random.rand()/1000

    # Apply custom weights
    for (i, j, lag), value in weights_dict.items():
        mask = (df["prev_action"] == i) & (
            df["action"] == j) & (df["time_lag"] == lag)
        indices = np.where(mask)[0]

        if indices.size > 0:
            # Set weight for matching entries
            weight_matrix[indices, j] = value

    # Set updated weights
    model.layers[0].set_weights([weight_matrix, bias])

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True), metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    y_pred = model.predict(x_test)
    p = np.argmax(y_pred, axis=1)

    printMetrics(y_test, p)
    return model

if __name__ == '__main__':
    LAYERS = [
        [128, 'relu'],
        [64, 'sigmoid']
    ]
    EPOCHS = 100
    BATCH_SIZE = 32
    runMLP(LAYERS, EPOCHS, BATCH_SIZE, weight_adjustment=False)
