from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def prepare_for_prophet(df):
    """Prepare DataFrame for Prophet."""
    return df.reset_index().rename(columns={'Date': 'ds', 'Price': 'y'})

def forecast_prophet(df, forecast_days):
    """Forecast stock prices using Prophet."""
    df_prepared = prepare_for_prophet(df)
    model = Prophet(daily_seasonality=True)
    model.fit(df_prepared)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)[['ds', 'yhat']].set_index('ds').tail(forecast_days)
    forecast.rename(columns={'yhat': 'Price'}, inplace=True)
    return forecast, model

def evaluate_prophet(df, test_size=10):
    """Evaluate Prophet model on the last `test_size` days."""
    df_prepared = prepare_for_prophet(df)
    train, test = df_prepared[:-test_size], df_prepared[-test_size:]

    model = Prophet(daily_seasonality=True)
    model.fit(train)

    future = model.make_future_dataframe(periods=test_size)
    forecast = model.predict(future)[['ds', 'yhat']].tail(test_size).set_index('ds')
    test = test.set_index('ds')

    return compute_metrics(test['y'], forecast['yhat'])

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
