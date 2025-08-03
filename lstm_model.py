import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def compute_metrics(y_true_series, y_pred_series):
    """Compute MAE, RMSE, and MAPE between true and predicted values."""
    df = pd.concat([y_true_series, y_pred_series], axis=1).dropna()
    df.columns = ['true', 'pred']
    if df.empty:
        return np.nan, np.nan, np.nan
    mae = mean_absolute_error(df['true'], df['pred'])
    rmse = np.sqrt(mean_squared_error(df['true'], df['pred']))
    nonzero_true = df['true'] != 0
    mape = np.mean(np.abs((df['true'][nonzero_true] - df['pred'][nonzero_true]) / df['true'][nonzero_true])) * 100 if nonzero_true.sum() > 0 else np.nan
    return round(mae, 4), round(rmse, 4), round(mape, 4)

def create_lstm_model(input_shape, units=50):
    """Build and compile an LSTM model."""
    model = Sequential([
        LSTM(units, return_sequences=False, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_lstm_data(df, sequence_length=60):
    """Prepare data for LSTM (scaled sequences and labels)."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Price']])
    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaler

def forecast_lstm(df, forecast_days, seq_len=60, epochs=20, batch_size=16):
    """Forecast future stock prices using trained LSTM model."""
    X, y, scaler = prepare_lstm_data(df, sequence_length=seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = create_lstm_model((seq_len, 1))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # Recursive multi-step forecast
    last_seq = X[-1]
    forecast_scaled = []
    for _ in range(forecast_days):
        pred = model.predict(last_seq.reshape(1, seq_len, 1), verbose=0)[0][0]
        forecast_scaled.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)

    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
    forecast_index = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    return pd.DataFrame({'Price': forecast}, index=forecast_index), model

def evaluate_lstm(df, seq_len=60, epochs=20, batch_size=16, train_ratio=0.9):
    """Evaluate LSTM model performance."""
    X, y, scaler = prepare_lstm_data(df, sequence_length=seq_len)
    if len(X) < 1:
        return np.nan, np.nan, np.nan
    split = int(len(X) * train_ratio)
    X_train, y_train, X_test, y_test = X[:split], y[:split], X[split:], y[split:]

    model = create_lstm_model((seq_len, 1))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    preds = model.predict(X_test)
    y_pred = scaler.inverse_transform(preds)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    return compute_metrics(pd.Series(y_true.flatten()), pd.Series(y_pred.flatten()))
