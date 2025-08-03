import yfinance as yf
import pandas as pd

def fetch_data(ticker, years=5):
    df = yf.download(ticker, period=f'{years}y', interval='1d')[['Close']]
    df.dropna(inplace=True)
    df.columns = ['Price']
    df.index.name = 'Date'
    return df

def preprocess_data(df):
    df = df.asfreq('D')  # Fill missing days
    df['Price'] = df['Price'].ffill().bfill()
    df = df.dropna()
    return df
