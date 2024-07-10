import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
# import matplotlib.pyplot as plt
import joblib
import os

# Mixed precision
mixed_precision.set_global_policy('mixed_float16')

# Global scaler
scaler = MinMaxScaler(feature_range=(0, 1))

def load_and_preprocess(file_path, fit_scaler=False):
    chunks = pd.read_csv(file_path, header=None, names=['DateTime', 'Open', 'Close', 'High', 'Low', 'Status'], 
                         delimiter=';', chunksize=10000)
    data = pd.concat(chunks)
    data['DateTime'] = pd.to_datetime(data['DateTime'], format='%Y%m%d %H%M%S')
    data.set_index('DateTime', inplace=True)
    close_data = data[['Close']].values
    if fit_scaler:
        scaler.partial_fit(close_data)  # Use partial_fit for incremental fitting
    return scaler.transform(close_data)

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

def tf_data_generator(file_paths, seq_length, batch_size):
    def gen():
        for file_path in file_paths:
            data = load_and_preprocess(file_path)
            x, y = create_sequences(data, seq_length)
            yield x, y

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, seq_length, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        )
    ).unbatch().batch(batch_size)

    # Improve performance with parallel calls and prefetching
    dataset = dataset.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    
    return dataset


def build_model(seq_length):
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Estimate steps per epoch based on file sizes
def estimate_steps(file_paths, batch_size):
    total_size = sum(os.path.getsize(file) for file in file_paths)
    estimated_samples = total_size / 200  # Rough estimate, adjust based on your data
    return int(estimated_samples / batch_size)

# Usage
train_files = ['train1.csv', 'train2.csv', 'train3.csv']
test_files = ['test1.csv']
seq_length = 60
batch_size = 32

# Fit scaler on training data
for file in train_files:
    load_and_preprocess(file, fit_scaler=True)

train_dataset = tf_data_generator(train_files, seq_length, batch_size)
test_dataset = tf_data_generator(test_files, seq_length, batch_size)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model(seq_length)

# Callbacks for optimization
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Estimate steps per epoch
steps_per_epoch = estimate_steps(train_files, batch_size)
validation_steps = estimate_steps(test_files, batch_size)

model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    validation_data=test_dataset,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr]
)

# Save the model and scaler
model.save('gold_price_model.h5')
joblib.dump(scaler, 'gold_price_scaler.pkl')

# # Plot training history
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()