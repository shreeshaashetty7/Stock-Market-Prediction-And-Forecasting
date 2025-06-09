import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load Data
def load_data_yahoo(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty and ".NS" in ticker:
            st.warning(" NSE ticker not available. Trying BSE fallback...")
            ticker = ticker.replace(".NS", ".BO")
            data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data found. Check the ticker or date range.")
        return data
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Preprocess Data
def preprocess_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    x, y = [], []
    for i in range(look_back, len(scaled_data)):
        x.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(x), np.array(y), scaler

# Train-Test Split
def split_data(x, y, train_ratio=0.8):
    train_size = int(len(x) * train_ratio)
    return x[:train_size], x[train_size:], y[:train_size], y[train_size:]

# LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(100, return_sequences=False),
        Dropout(0.3),
        Dense(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def train_lstm(x_train, y_train, x_test, scaler):
    x_train_lstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test_lstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    model = build_lstm_model((x_train_lstm.shape[1], 1))
    model.fit(x_train_lstm, y_train, epochs=10, batch_size=32, verbose=0)
    predictions = model.predict(x_test_lstm)
    return model, scaler.inverse_transform(predictions)

# XGBoost
def train_xgboost(x_train, y_train, x_test, scaler):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return model, scaler.inverse_transform(predictions.reshape(-1, 1))

# ARIMA
def train_arima(data, days):
    model = ARIMA(data['Close'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)
    return forecast

# Aggregate Predictions
def aggregate_predictions(lstm_pred, xgb_pred, arima_pred, days):
    lstm_pred = lstm_pred.flatten()[:days]
    xgb_pred = xgb_pred.flatten()[:days]
    arima_pred = arima_pred.values[:days]
    combined_pred = (lstm_pred + xgb_pred + arima_pred) / 3
    return combined_pred

# Sentiment Score
def get_sentiment_score(ticker):
    try:
        news_url = f"https://www.google.com/search?q={ticker}+stock+news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(news_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [h.text for h in soup.find_all('h3') if h.text != '']
        if not headlines:
            return 0.0
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
        avg_score = np.mean(scores)
        return avg_score
    except:
        return 0.0

# Recommendation
def generate_recommendation(last_close, predicted_price, sentiment_score):
    last_close = float(last_close)
    predicted_price = float(predicted_price)
    change = (predicted_price - last_close) / last_close * 100

    if sentiment_score > 0.3:
        if change > 2:
            return "BUY", "Strong positive sentiment and upward trend expected. Consider buying."
        elif change < -2:
            return "HOLD", "Mixed signals. Despite positive sentiment, price may drop."
        else:
            return "HOLD", "Positive sentiment but minimal price change. Hold your position."
    elif sentiment_score < -0.3:
        if change < -2:
            return "SELL", "Negative sentiment and falling trend. Consider selling."
        elif change > 2:
            return "HOLD", "Price may rise but sentiment is negative. Be cautious."
        else:
            return "HOLD", "Negative sentiment with low price movement. Hold advised."
    else:
        if change > 2:
            return "BUY", "Neutral sentiment but price expected to rise."
        elif change < -2:
            return "SELL", "Neutral sentiment but price expected to drop."
        else:
            return "HOLD", "Minimal price change. Holding position recommended."

# Future Dates
def get_future_dates(last_date, num_days):
    future_dates = []
    current_date = last_date
    for _ in range(num_days):
        next_date = current_date + timedelta(days=1)
        while next_date.weekday() > 4:
            next_date += timedelta(days=1)
        future_dates.append(next_date.strftime('%Y-%m-%d'))
        current_date = next_date
    return future_dates

# Streamlit UI
def main():
    st.set_page_config(page_title="Stock Market Prediction", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
            body {
                background-color: #0e1117;
                color: white;
            }
            .stButton>button {
                color: white;
                background-color: #4CAF50;
                border-radius: 12px;
                padding: 10px 20px;
                font-size: 16px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title(" Stock Market Prediction & Recommendation Engine")
    ticker = st.text_input("Enter stock ticker (e.g., TCS.NS): ").upper()
    num_days = st.slider("Select prediction range (days):", min_value=1, max_value=365, value=3)

    if st.button(" Predict") and ticker:
        start_date = '2015-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        data = load_data_yahoo(ticker, start_date, end_date)
        if data is None:
            return

        x, y, scaler = preprocess_data(data)
        x_train, x_test, y_train, y_test = split_data(x, y)

        lstm_model, lstm_pred = train_lstm(x_train, y_train, x_test, scaler)
        xgb_model, xgb_pred = train_xgboost(x_train, y_train, x_test, scaler)
        arima_pred = train_arima(data, days=num_days)

        aggregated_predictions = aggregate_predictions(lstm_pred, xgb_pred, arima_pred, days=num_days)

        last_date = data.index[-1]
        future_dates = get_future_dates(last_date, num_days)

        st.subheader(" Predicted Prices:")
        prediction_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': aggregated_predictions.flatten()
        })
        st.dataframe(prediction_df.style.highlight_max(axis=0))

        last_close = data['Close'].iloc[-1]
        final_prediction = aggregated_predictions[-1]

        sentiment_score = get_sentiment_score(ticker)
        st.subheader(" Sentiment Analysis")
        st.metric(label="Sentiment Score", value=f"{sentiment_score:.2f}")
        st.progress(int((sentiment_score + 1) * 50))

        decision, guidance = generate_recommendation(last_close, final_prediction, sentiment_score)

        st.subheader(f" Investment Recommendation: {decision}")
        st.write(guidance)

if __name__ == "__main__":
    main()
