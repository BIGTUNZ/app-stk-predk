

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Title for the web app
st.title('Apple Stock Price Prediction')

# Load the trained LSTM model
try:
    model_lstm = tf.keras.models.load_model("lstm_model.h5")
    st.write("LSTM model loaded successfully!")
except Exception as e:
    st.error(f"Error loading LSTM model: {e}")

# Load the scaler
try:
    scaler = joblib.load("scaler.pkl")
    st.write("Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading scaler: {e}")

# Input features from the user
st.header('Input Features')

# Feature inputs based on stock prediction project
open_price = st.number_input('Opening Price', min_value=0.0, step=0.01, value=150.0)
high_price = st.number_input('Highest Price of the Day', min_value=0.0, step=0.01, value=155.0)
low_price = st.number_input('Lowest Price of the Day', min_value=0.0, step=0.01, value=145.0)
volume = st.number_input('Volume of Stocks Traded', min_value=0, step=1, value=1000000)

# Create a dictionary for the model input
input_data = {
    'Volume': [volume],
    'Open': [open_price],
    'High': [high_price],
    'Low': [low_price]
}

# Convert input data to dataframe
input_df = pd.DataFrame(input_data)

# Display the input data
st.subheader('Input Data')
st.write(input_df)

# Prediction
if st.button('Predict'):
    try:
         # Add a dummy 'Close' column with a placeholder value
        input_df['Close'] = 0

        # Reorder columns to match the expected feature order
        input_df = input_df[['Close', 'Volume', 'Open', 'High', 'Low']]
        
        # Scale input data
        scaled_data = scaler.transform(input_df)

        # Predict using the LSTM model
        predictions = model_lstm.predict(scaled_data)

        # Post-process predictions (e.g., inverse scaling)
        extended_predictions = np.zeros((predictions.shape[0], scaler.n_features_in_))
        extended_predictions[:, 0] = predictions.flatten()
        final_predictions = scaler.inverse_transform(extended_predictions)[:, 0]

        # Display the predicted price
        st.success(f"The predicted closing price of Apple stock is ${final_predictions[0]:.2f}.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

