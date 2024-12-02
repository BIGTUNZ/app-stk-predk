# Apple Stock Price Prediction

**Predicting Apple Stock Prices Using Machine Learning and Deep Learning**

This project leverages historical stock data to predict future stock prices using both traditional machine learning models (Random Forest) and advanced deep learning models (LSTM). It demonstrates expertise in data preprocessing, feature engineering, and model evaluation with real-world financial data.

---

##  Table of Contents
- [About the Project](##About-the-project)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Model Performance](#model-performance)
- [Getting Started](#getting-started)
- [Future Enhancements](#future-enhancements)
- [Contact](#contact)

---

##  About the Project

This project focuses on predicting Apple's stock prices using historical data. The dataset includes features such as opening, closing, high, and low prices, trading volume, and moving averages. The goal is to explore the predictive power of both machine learning and deep learning models.

### Objectives:
1. Use a **Random Forest model** to evaluate the relationship between stock price features and predict outcomes.
2. Build an **LSTM (Long Short-Term Memory) network**, a type of Recurrent Neural Network, to predict the stock's closing prices.
3. Compare and evaluate the performance of these models.
4. Deploy using streamlit
---

##  Key Features
- **Data Collection**: Automated data retrieval using `yfinance`.
-  **Data Preprocessing**: 
  - Feature scaling using `MinMaxScaler`.
  - Computation of moving averages (50-day and 200-day).
-  **Machine Learning Model**: Random Forest for initial predictions.
-  **Deep Learning Model**: LSTM network to capture temporal dependencies in stock price movements.
-  **Visualizations**:
  - Historical stock price trends.
  - Model predictions vs. actual prices.

---
##  Model Performance

### Evaluation Metrics:
- **Mean Squared Error (MSE)**: 47.63
- **Mean Absolute Error (MAE)**: 5.45
- **R-Squared (R²)**: 0.94

### Key Insights:
- The **Random Forest model** provides a good baseline with reasonable accuracy.
- The **LSTM model** captures temporal dependencies, producing better predictions for sequential data like stock prices.

---

##  Getting Started

### Prerequisites:
- Install Python 3.8+
- Clone the repository and install required libraries:

```bash
git clone https://github.com/yourusername/apple-stock-prediction.git
cd apple-stock-prediction
pip install -r requirements.txt
```

### Running the Code:
1. **Download the dataset**: Automatically fetches using `yfinance`.
2. **Preprocess the data**: Scales and transforms features for model compatibility.
3. **Train and Evaluate Models**:
   ```bash
   python train_random_forest.py
   python train_lstm.py
   ```
4. **Visualize Results**: Use Jupyter Notebook for step-by-step analysis.

---

##  Future Enhancements
- Incorporate **sentiment analysis** of news articles to improve predictions.
- Experiment with **other deep learning architectures**, such as GRUs.
- Add real-time stock price prediction capability.

---

##  Contact

Feel free to reach out for collaboration or queries:

- **Name**: Yusuf Babatunde Jubril
- **Email**: jubriltunde643@gmail.com
- **GitHub**: https://github.com/BIGTUNZ/
- **LinkedIn**: https://www.linkedin.com/in/babatunde-yusuf-063772278/
