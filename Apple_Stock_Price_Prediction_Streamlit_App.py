

# Import necessary libraries
import streamlit as st
import pandas as pd

import joblib

# Title for the web app
st.title('Apple Stock Price Prediction')

# Load the trained model
model = joblib.load("scaler.pkl")

# Input features from the user
st.header('Input Features')

# Feature inputs based on stock prediction project
open_price = st.number_input('Opening Price', min_value=0.0, step=0.01, value=150.0)
high_price = st.number_input('Highest Price of the Day', min_value=0.0, step=0.01, value=155.0)
low_price = st.number_input('Lowest Price of the Day', min_value=0.0, step=0.01, value=145.0)
volume = st.number_input('Volume of Stocks Traded', min_value=0, step=1, value=1000000)

# Create a dictionary for the model input
input_data = {
    'Close': [close_price],
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
    prediction = model.predict(input_df)
    st.success(f"The predicted closing price of Apple stock is ${prediction[0]:.2f}.")
    
