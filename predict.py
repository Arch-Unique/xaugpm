# predict.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import time

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, header=None, names=['DateTime', 'Open', 'Close', 'High', 'Low', 'Status'], delimiter=';')
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S')
    df.set_index('DateTime', inplace=True)
    return df[['Close']]

def load_gold_price_model():
    model = load_model('gold_price_model.h5')
    scaler = joblib.load('gold_price_scaler.pkl')
    return model, scaler

def predict_next_hour(model, scaler, input_data):
    # Ensure input_data is a numpy array with shape (n, 1) where n can vary
    if isinstance(input_data, pd.DataFrame):
        input_data = input_data.values
    
    if len(input_data.shape) != 2 or input_data.shape[1] != 1:
        raise ValueError("Input data should have shape (n, 1) where n is the number of time steps")
    
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    
    predictions = []
    current_sequence = scaled_input.reshape(1, -1, 1)
    
    for _ in range(60):  # Predict next 60 minutes
        next_pred = model.predict(current_sequence)
        predictions.append(next_pred[0, 0])
        current_sequence = np.concatenate([current_sequence, next_pred.reshape(1, 1, 1)], axis=1)
    
    # Inverse transform the predictions
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)

# Example usage
def predict_gold_price(file_path):
    input_data = load_and_preprocess(file_path)
    close_prices = input_data['Close'].values.reshape(-1, 1)
    return predict_next_hour(model, scaler, close_prices)

def simulate_new_data(last_data, num_minutes):
    """Simulate new data for testing purposes"""
    last_price = last_data['Close'].iloc[-1]
    new_index = pd.date_range(start=last_data.index[-1] + pd.Timedelta(minutes=1), periods=num_minutes, freq='1T')
    new_prices = last_price + np.random.randn(num_minutes) * 0.1  # Simple random walk
    return pd.DataFrame({'Close': new_prices}, index=new_index)

def update_model(model, scaler, new_data):
    """
    Update the model with new data.
    
    :param model: The trained Keras model
    :param scaler: The fitted MinMaxScaler
    :param new_data: DataFrame with 'Close' prices for the last 5 minutes
    """
    # Ensure new_data is in the correct format
    if not isinstance(new_data, pd.DataFrame) or 'Close' not in new_data.columns:
        raise ValueError("new_data should be a DataFrame with a 'Close' column")
    scaled_data = scaler.transform(new_data[['Close']])
    X, y = create_sequences(scaled_data, seq_length=60)
    if len(X) == 0:
        print("Not enough new data to update the model")
        return model, scaler
    
    # Update the model
    model.fit(X, y, epochs=1, batch_size=1, verbose=0)
    return model, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Update Continuously
def continuous_predict_and_update(initial_data, update_interval=300, prediction_horizon=60):
    """
    Continuously predict and update the model.
    
    :param initial_data: DataFrame with initial 'Close' prices
    :param update_interval: Time in seconds between updates (default: 300 seconds = 5 minutes)
    :param prediction_horizon: Number of minutes to predict into the future (default: 60 minutes)
    """
    model, scaler = load_gold_price_model()
    
    while True:
        # Make prediction
        current_data = initial_data['Close'].values[-60:].reshape(-1, 1)
        prediction = predict_next_hour(model, scaler, current_data)
        
        print(f"Prediction for the next hour: {prediction.flatten()}")
        
        # Wait for new data
        time.sleep(update_interval)
        
        # In a real scenario, you would fetch new data here
        # For this example, we'll simulate new data
        new_data = simulate_new_data(initial_data, 5)  # 5 minutes of new data
        
        # Update the model
        model, scaler = update_model(model, scaler, new_data)
        
        # Update initial_data for the next iteration
        initial_data = pd.concat([initial_data, new_data])
        
        # Optional: Save updated model and scaler
        model.save('updated_gold_price_model.h5')
        joblib.dump(scaler, 'updated_gold_price_scaler.pkl')

# Load the model and scaler
# initial_data = load_and_preprocess('initial.csv')
# continuous_predict_and_update(initial_data)