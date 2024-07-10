import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense,Bidirectional
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.callbacks import EarlyStopping


# Function to load and preprocess data
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, header=None, names=['DateTime', 'Open', 'Close', 'High', 'Low', 'Status'], delimiter=';')
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S')
    df.set_index('DateTime', inplace=True)
    return df[['Close']]

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


# Load the datasets
print("Loading Datasets")
train1 = load_and_preprocess('train1.csv')
train2 = load_and_preprocess('train2.csv')
train3 = load_and_preprocess('train3.csv')
test = load_and_preprocess('test1.csv')

# Combine training data
train_data = pd.concat([train1, train2, train3])

print("Normalize Datasets")

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test)

# Create Sequences
seq_length = 60  # 1 hour of data (60 minutes)
X_train, y_train = create_sequences(scaled_train, seq_length)
X_test, y_test = create_sequences(scaled_test, seq_length)

print("Build Model")
# Build the LSTM model
model = Sequential([
    Bidirectional(LSTM(50, activation='relu', return_sequences=True)),
    Bidirectional(LSTM(50, activation='relu')),
    Dense(1)
])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

print("Fitting Model")
model.fit(X_train, y_train, 
          epochs=100,  # Set this to a large number
          batch_size=32, 
          validation_split=0.1, 
          callbacks=[early_stopping],
          verbose=1)

#model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Save the model and scaler
model.save('gold_price_model.h5')
joblib.dump(scaler, 'gold_price_scaler.pkl')

# Function to predict the next hour
def predict_next_hour(model, scaler, input_data):
    # Ensure input_data is a numpy array with shape (60, 1)
    if isinstance(input_data, pd.DataFrame):
        input_data = input_data.values
    
    if input_data.shape != (60, 1):
        raise ValueError("Input data should have shape (60, 1)")
    
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    
    predictions = []
    current_sequence = scaled_input.reshape(1, 60, 1)
    
    for _ in range(60):  # Predict next 60 minutes
        next_pred = model.predict(current_sequence)
        predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred[0, 0]
    
    # Inverse transform the predictions
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)

from google.colab import files
files.download('gold_price_model.h5')
files.download('gold_price_scaler.pkl')

# Example usage
# last_60_minutes = test['Close'].iloc[-60:].values.reshape(-1, 1)
# next_hour = predict_next_hour(model, scaler, last_60_minutes)
# print(next_hour)


# Get the last sequence from the test data
# last_sequence = scaled_test[-seq_length:]

# # Predict the next hour
# next_hour_predictions = predict_next_hour(model, last_sequence)

# # Inverse transform the predictions
# next_hour_predictions = scaler.inverse_transform(next_hour_predictions)

# # Create a DataFrame for the next hour predictions
# last_datetime = test.index[-1]
# next_hour_datetimes = pd.date_range(start=last_datetime + pd.Timedelta(minutes=1), periods=60, freq='1T')
# next_hour_df = pd.DataFrame(next_hour_predictions, index=next_hour_datetimes, columns=['Predicted_Price'])

# # Plot the next hour predictions
# plt.figure(figsize=(12, 6))
# plt.plot(test.index[-60:], test['Close'].iloc[-60:], label='Actual')
# plt.plot(next_hour_df.index, next_hour_df['Predicted_Price'], label='Next Hour Prediction')
# plt.title('Gold/USD Price Prediction for the Next Hour')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

# print(next_hour_df)