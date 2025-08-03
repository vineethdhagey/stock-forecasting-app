import streamlit as st
import matplotlib.pyplot as plt
from data_utils import fetch_data, preprocess_data


import prophet_model
import arima_model
import sarima_model
import lstm_model

import streamlit as st
import base64

st.set_page_config(page_title="📊 Stock Forecasting App", layout="wide")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call the function with your local image filename
set_background('b.jpg')




st.title("📈 Stock Price Forecast Dashboard")
st.markdown("<h3 style='color:#333;'>All prices are shown in <b>**USD**</b></h3>", unsafe_allow_html=True)


ticker = st.selectbox("Choose Stock", ['AAPL', 'TSLA', 'JPM'])
model_name = st.selectbox("Choose Model", ['ARIMA', 'SARIMA', 'Prophet', 'LSTM'])
forecast_days = st.slider("Select Forecast Horizon", 7, 14, 7)

# Load data
df = fetch_data(ticker)

# Model Dispatch
if model_name == 'Prophet':
    forecast, model = prophet_model.forecast_prophet(df, forecast_days)
    mae, rmse, mape = prophet_model.evaluate_prophet(df)
elif model_name == 'ARIMA':
    forecast, model = arima_model.forecast_arima(df, forecast_days)
    mae, rmse, mape = arima_model.evaluate_arima(df)
elif model_name == 'SARIMA':
    forecast, model = sarima_model.forecast_sarima(df, forecast_days)
    mae, rmse, mape = sarima_model.evaluate_sarima(df)
else:  # LSTM
    forecast, model = lstm_model.forecast_lstm(df, forecast_days)
    mae, rmse, mape = lstm_model.evaluate_lstm(df)

# Display Forecast Plot
st.subheader(f"{ticker} Forecast using {model_name}")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['Price'].tail(30), label="Last 30 Days", color='blue')
ax.plot(forecast['Price'], label=f"Next {forecast_days} Days", color='green')

ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title(f"{ticker} Price Forecast")
ax.legend()
st.pyplot(fig)

# Display Metrics
st.subheader("📊 Evaluation Metrics")
st.metric("MAE", f"{mae:.2f}")
st.metric("RMSE", f"{rmse:.2f}")
st.metric("MAPE", f"{mape:.2f}%")
