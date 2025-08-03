from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def forecast_arima(df, forecast_days):
    model = ARIMA(df['Price'], order=(5,1,0))
    fitted = model.fit()
    forecast = fitted.forecast(steps=forecast_days)
    forecast.index = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    return forecast.to_frame(name='Price'), fitted

def evaluate_arima(df):
    train = df[:-10]
    test = df[-10:]

    model = ARIMA(train, order=(5,1,0))
    fitted = model.fit()
    forecast = fitted.forecast(steps=10)
    forecast.index = test.index  # Important: match date index

    y_true = test['Price']
    y_pred = forecast

    return compute_metrics(y_true, y_pred)


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(y_true_series, y_pred_series):
    # Combine and align both series on datetime index
    df = pd.concat([y_true_series, y_pred_series], axis=1)
    df.columns = ['true', 'pred']
    df.dropna(inplace=True)

    if df.empty:
        return np.nan, np.nan, np.nan

    mae = mean_absolute_error(df['true'], df['pred'])
    rmse = np.sqrt(mean_squared_error(df['true'], df['pred']))

    # Avoid divide-by-zero in MAPE
    nonzero_true = df['true'] != 0
    if nonzero_true.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((df['true'][nonzero_true] - df['pred'][nonzero_true]) / df['true'][nonzero_true])) * 100

    return round(mae, 4), round(rmse, 4), round(mape, 4)

