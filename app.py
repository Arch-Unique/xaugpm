import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import time
from predict import load_and_preprocess, predict_next_hour, simulate_new_data, update_model, load_gold_price_model

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model, st.session_state.scaler = load_gold_price_model()
if 'initial_data' not in st.session_state:
    st.session_state.initial_data = load_and_preprocess('initial.csv')
if 'predictions' not in st.session_state:
    st.session_state.predictions = pd.DataFrame(columns=['DateTime', 'Predicted'])
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# Streamlit app
st.title('Gold Price Prediction')

# Plot actual vs predicted prices
def plot_actual_vs_predicted():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(st.session_state.initial_data.index, st.session_state.initial_data['Close'], label='Actual')
    ax.plot(st.session_state.predictions['DateTime'], st.session_state.predictions['Predicted'], label='Predicted')
    ax.set_title('Actual vs Predicted Gold Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

# Plot 1-hour prediction
def plot_hour_prediction(prediction):
    fig, ax = plt.subplots(figsize=(12, 6))
    minutes = range(60)
    ax.plot(minutes, prediction)
    ax.set_title('1-Hour Price Prediction')
    ax.set_xlabel('Minutes')
    ax.set_ylabel('Predicted Price')
    st.pyplot(fig)

# Continuous predict and update function
def continuous_predict_and_update():
    current_time = time.time()
    if current_time - st.session_state.last_update >= 300:  # 5 minutes
        # Make prediction
        current_data = st.session_state.initial_data['Close'].values[-60:].reshape(-1, 1)
        prediction = predict_next_hour(st.session_state.model, st.session_state.scaler, current_data)
        
        # Update predictions DataFrame
        new_predictions = pd.DataFrame({
            'DateTime': pd.date_range(start=st.session_state.initial_data.index[-1], periods=60, freq='1T'),
            'Predicted': prediction.flatten()
        })
        st.session_state.predictions = pd.concat([st.session_state.predictions, new_predictions])
        
        # Simulate new data (replace this with actual data fetching in production)
        new_data = simulate_new_data(st.session_state.initial_data, 5)
        
        # Update model
        st.session_state.model, st.session_state.scaler = update_model(st.session_state.model, st.session_state.scaler, new_data)
        
        # Update initial data
        st.session_state.initial_data = pd.concat([st.session_state.initial_data, new_data])
        
        # Update last update time
        st.session_state.last_update = current_time
        
        # Save updated model and scaler
        st.session_state.model.save('updated_gold_price_model.h5')
        joblib.dump(st.session_state.scaler, 'updated_gold_price_scaler.pkl')

# Main app logic
continuous_predict_and_update()

# Display plots
st.subheader('Actual vs Predicted Prices Over Time')
plot_actual_vs_predicted()

st.subheader('1-Hour Price Prediction')
latest_prediction = st.session_state.predictions.iloc[-60:]['Predicted'].values
plot_hour_prediction(latest_prediction)

st.subheader('Predicted Prices for the Next Hour')
st.write(pd.DataFrame({
    'Minute': range(1, 61),
    'Predicted Price': latest_prediction
}))

# Add a refresh button
if st.button('Refresh'):
    st.experimental_rerun()

# Display last update time
st.write(f"Last updated: {pd.to_datetime(st.session_state.last_update, unit='s')}")

# Auto-refresh every 5 minutes
if time.time() - st.session_state.last_update >= 300:
    st.experimental_rerun()