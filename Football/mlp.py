
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from common import *
from sklearn.model_selection import train_test_split

LAYERS = [
    [128, 'relu']
    # [64, 'tanh'],
    # [32, 'sigmoid']
]
OPTIMISER = 'adam'
LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
METRICS = ['accuracy']
EPOCHS = 100
BATCH_SIZE = 32

df = pd.read_csv(f'matches/{match_folder}/match_data_gat.csv')
action_df = pd.read_csv('fkeys/action.csv')
action_map = {action_df['action'][i]: action_df['id'][i]
              for i in range(len(action_df))}

# Load PCMCI link file
link_file = pd.read_csv(f"matches/{match_folder}/aggregated_links.csv")

weights_dict = {}
for _, row in link_file.iterrows():
    i, j, lag, link_value = row["Variable i"], row["Variable j"], row["Time lag of i"], row["Link value"]
    weights_dict[(i, j, lag)] = link_value

# Extract necessary columns
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]  # Target action

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(
    shape=(X.shape[1],)))
for units, activation in LAYERS:
    model.add(tf.keras.layers.Dense(units, activation=activation))
model.add(tf.keras.layers.Dense(len(action_map)))


model.build(input_shape=(None, X.shape[1]))

# Initialise weights for the first layer (prev_action → action mapping)
first_layer_weights = model.layers[0].get_weights()  # Get default weights
weight_matrix, bias = first_layer_weights

# Apply custom weights based on the links file
for (i, j, lag), value in weights_dict.items():
    mask = (df["prev_action"] == i) & (
        df["action"] == j) & (df["time_lag"] == lag)
    indices = np.where(mask)[0]

    if indices.size > 0:
        weight_matrix[indices, j] = value  # Set weight for matching entries

# Set updated weights
model.layers[0].set_weights([weight_matrix, bias])

model.compile(optimizer=OPTIMISER, loss=LOSS, metrics=METRICS)
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
y_pred = model.predict(x_test)
p = np.argmax(y_pred, axis=1)

# Metrics
accuracy = accuracy_score(y_test, p)
precision = precision_score(y_test, p, average='weighted')
recall = recall_score(y_test, p, average='weighted')
f1 = f1_score(y_test, p, average='weighted')
print(
    f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
# Accuracy: 0.6950, Precision: 0.6597, Recall: 0.6950, F1-score: 0.6369 (with weight adjustment)
# Accuracy: 0.6450, Precision: 0.5869, Recall: 0.6450, F1-score: 0.5708 (no weight adjustment)
