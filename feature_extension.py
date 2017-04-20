import pandas as pd
import pickle
import numpy as np


def add_feature_industry():
    # Adds information about which industry particular company belongs
    # For industry selection GICS Sector is used
    # Source https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
    with open("sp25tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    for ticker in tickers:
        if ticker in ["BAC", "BLK", "C", "MS", "SCHW", "WFC", "ZION"]:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker), index_col=0)
            df['Industry'] = "Financials"
            df['Industry_ID'] = 1
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        if ticker in ["AMZN","AZO", "CMG", "PCLN"]:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker), index_col=0)
            df['Industry'] = "Consumer Discretionary"
            df['Industry_ID'] = 2
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        if ticker in ["AAPL", "CSCO", "GOOG", "INTC", "MA", "MSFT", "NVDA"]:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker), index_col=0)
            df['Industry'] = "Information Technology"
            df['Industry_ID'] = 3
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        if ticker in ["DAL", "GE"]:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker), index_col=0)
            df['Industry'] = "Industrials"
            df['Industry_ID'] = 4
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        if ticker in ["EXC"]:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker), index_col=0)
            df['Industry'] = "Utilities"
            df['Industry_ID'] = 5
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        if ticker in ["ISRG", "PFE"]:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker), index_col=0)
            df['Industry'] = "Health Care"
            df['Industry_ID'] = 6
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        if ticker in ["T"]:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker), index_col=0)
            df['Industry'] = "Telecommunication Services"
            df['Industry_ID'] = 7
            df.to_csv('stock_dfs/{}.csv'.format(ticker))


def calc_beta(df):
    np_array = df.values
    s = np_array[:, 0]
    m = np_array[:, 1]

    covariance = np.cov(s, m)
    beta = covariance[0, 1]/covariance[1, 1]
    return beta


def rolling_apply(df, period, func, min_periods=None):
    if min_periods is None:
        min_periods = period
    result = pd.Series(np.nan, index=df.index)

    for i in range(1, len(df)+1):
        sub_df = df.iloc[max(i-period, 0):i, :]
        if len(sub_df) >= min_periods:
            idx = sub_df.index[-1]
            result[idx] = func(sub_df)

    return result


def add_beta(period=25):
    with open("sp25tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    df = pd.DataFrame()
    df_index = pd.read_csv('stock_dfs/SPX.csv', index_col=0)
    df['Market'] = (df_index['Open'] - df_index['Open'].mean()) / (df_index['Open'].max() - df_index['Open'].min())
    df['Market'].fillna(method='pad', inplace=True)
    df['Market'].fillna(method='bfill', inplace=True)
    for ticker in tickers:
        df_stock = pd.read_csv('stock_dfs/{}.csv'.format(ticker), index_col=0)
        df['Stock'] = (df_stock['Open'] - df_stock['Open'].mean()) / (df_stock['Open'].max() - df_stock['Open'].min())
        beta = rolling_apply(df, period=period, func=calc_beta, min_periods=3)
        beta.name = 'Beta'
        df_stock['Beta'] = beta
        df_stock['Beta'].fillna(method='pad', inplace=True)
        df_stock['Beta'].fillna(method='bfill', inplace=True)
        df_stock.to_csv('stock_dfs/{}.csv'.format(ticker))