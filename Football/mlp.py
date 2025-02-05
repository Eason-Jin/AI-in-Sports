
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
        if os.path.exists(f'matches/{match_folder}/match_data_gat.csv'):
            print(f'Processing {match_folder}')
            df_list.append(pd.read_csv(
                f'matches/{match_folder}/match_data_gat.csv'))
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

    if weight_adjustment:
        model.build(input_shape=(None, X.shape[1]))

        # Initialise weights for the first layer (prev_action → action mapping)
        # Get default weights
        first_layer_weights = model.layers[0].get_weights()
        weight_matrix, bias = first_layer_weights

        # Load PCMCI link file
        link_file = pd.read_csv(f"aggregated_links.csv")

        weights_dict = {}
        for _, row in link_file.iterrows():
            i, j, lag, link_value = row["Variable i"], row["Variable j"], row["Time lag of i"], row["Link value"]
            weights_dict[(i, j, lag)] = link_value

        # Apply custom weights based on the links file
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


# [128, relu], [64, sigmoid]
# Accuracy: 0.7200, Precision: 0.6779, Recall: 0.7200, F1-score: 0.6648 (with weight adjustment)
# Accuracy: 0.6900, Precision: 0.6685, Recall: 0.6900, F1-score: 0.6368 (no weight adjustment)

# [256, relu], [128, sigmoid]
# Accuracy: 0.8250, Precision: 0.7956, Recall: 0.8250, F1-score: 0.7942 (with weight adjustment)
# No weight adjustment yields similar results

if __name__ == '__main__':
    LAYERS = [
        [64, 'leaky_relu'],
        [32, 'softmax']
    ]
    EPOCHS = 100
    BATCH_SIZE = 32
    runMLP(LAYERS, EPOCHS, BATCH_SIZE, weight_adjustment=True)
