import pandas as pd
import pickle


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
    return df



# working ...
def add_features_locality(df, size):
    min_l = df['Low'].shift(-1)
    max_l = df['High'].shift(-1)
    for i in range(1, size):
        if df['Low'].shift(-i) < min_l:
            min_l = df['Low']
        if df['High'].shift(-i) < max_l:
            max_l = df['High']
    df['Feature_5'] = (df['Open'] - df['Open'].shift(-1))/(max_l - min_l)











