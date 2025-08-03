# 📈 Stock Price Forecast Dashboard
An interactive Streamlit web app for forecasting stock prices using multiple models: ARIMA, SARIMA, Prophet, and LSTM.
The app fetches live historical data using Yahoo Finance, performs preprocessing, trains models, and provides forecasted stock prices along with evaluation metrics (MAE, RMSE, MAPE).

# 🚀 Features
**Multi-model support:** Choose between ARIMA, SARIMA, Prophet, and LSTM for forecasting.

**Forecast horizon:** Predict stock prices for the next 7–14 days.

**Interactive Interface:** Built with Streamlit for an easy and smooth user experience.

**Evaluation Metrics:** View model performance with MAE, RMSE, and MAPE.

**Visual Forecasts:** Last 30 days of actual data + future predictions.

**Data Source:** Pulls 5 years of daily data directly from Yahoo Finance.

**Stocks included:**

1)Apple (AAPL)

2)Tesla (TSLA)

3)JPMorgan (JPM)

**Currency:** All prices are displayed in USD.

# 🛠️ Tech Stack
**Frontend:** Streamlit

**Models:**

1) ARIMA & SARIMA: statsmodels

2) Prophet: fbprophet (Prophet by Meta)

3) LSTM: TensorFlow/Keras

**Data:** yfinance for historical stock data

**Visualization:** matplotlib

# 🔄 How It Works
**Data Collection:**

The app automatically fetches 5 years of historical daily stock price data from Yahoo Finance.

**Preprocessing:**

Missing values are filled, and data is resampled to a daily frequency.

**Model Training:**

The selected model (ARIMA/SARIMA/Prophet/LSTM) is trained using this 5-year dataset.

**Forecasting:**

Forecasts for the next 7–14 days are generated and displayed.

**Evaluation:**

Key metrics (MAE, RMSE, MAPE) are calculated to assess prediction accuracy.


## 📁 Project Structure

```

stock market app/
│
├── app.py # Main Streamlit app
├── data_utils.py # Data fetching & preprocessing
├── prophet_model.py # Prophet forecasting model
├── lstm_model.py # LSTM forecasting model
├── arima_model.py # ARIMA forecasting model
├── sarima_model.py # SARIMA forecasting model
├── b.jpg # Background image
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── time/ # Virtual environment

```

---





# ⚙️ Installation & Setup
**1) Clone the repository:**
 
 ```bash
 git clone https://github.com/dvinzzzz/stock-forecasting-app.git
 cd stock-forecasting-app
```

2) Create and activate a virtual environment

    Windows:
   python -m venv venv

   venv\Scripts\activate
   
4) Install dependencies

   pip install -r requirements.txt
   
5) Run the app

   streamlit run app.py
   
# 🖼️ Screenshots

<img width="1480" height="412" alt="Screenshot 2025-08-03 182930" src="https://github.com/user-attachments/assets/1b2d6a68-a81f-42d8-90dc-e2a6b7ab1d43" />
<img width="1817" height="795" alt="Screenshot 2025-08-03 182912" src="https://github.com/user-attachments/assets/63c041ed-8250-4608-9c9a-e4e3d7454f23" />
<img width="1817" height="569" alt="Screenshot 2025-08-03 182807" src="https://github.com/user-attachments/assets/c08c4cee-b417-4c5c-8e75-697b659c6ad2" />




# 📊 Models Used
ARIMA & SARIMA: Statistical time-series models for short-term forecasting.

Prophet: Robust model by Meta for seasonality & trend detection.

LSTM: Deep learning model for sequence prediction with memory.

# 📄 License
This project is licensed under the MIT License.






