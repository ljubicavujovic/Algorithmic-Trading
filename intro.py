import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import re
import pandas as pd
import pickle
import requests
import csv
import model_all
import feature_extension as fe
def select_tickers():
    # Read ticker list
    selected_tickers = []
    with open("selected_tickers.txt", "r") as f:
        for ticker in f:
            selected_tickers.append(ticker.rstrip())
    with open("sp25tickers.pickle", "wb") as f:
        pickle.dump(selected_tickers, f)
    return selected_tickers


def get_google_finance_intraday(ticker, period=60, days=1):
    # Get OHLC data from Google Finance for specified time bucket
    uri = 'http://www.google.com/finance/getprices' \
          '?i={period}&p={days}d&f=d,o,h,l,c,v&df=cpct&q={ticker}'.format(ticker=ticker, period=period, days=days)
    page = requests.get(uri)
    reader = csv.reader(page.content.splitlines())
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    rows = []
    times = []
    for row in reader:
        if re.match('^[a\d]', row[0]):
            if row[0].startswith('a'):
                start = dt.datetime.fromtimestamp(int(row[0][1:]))
                times.append(start)
            else:
                times.append(start+dt.timedelta(seconds=period*int(row[0])))
            rows.append(map(float, row[1:]))
    if len(rows):
        return pd.DataFrame(rows, pd.DatetimeIndex(times, name='Date'), columns)
    else:
        return pd.DataFrame(rows, pd.DatetimeIndex(times, name='Date'))


def get_data(reload_sp25=False):
    # Get intraday data for selected companies
    if reload_sp25:
        tickers = select_tickers()
    else:
        with open("sp25tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    for ticker in tickers:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = get_google_finance_intraday(ticker)
            df = get_features(df)
            df.fillna(method='pad', inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


def get_features(df):
    df['Feature_1'] = (df['Open'] - df['Open'].shift(-1))/df['Open'].shift(-1)
    df['Feature_2'] = (df['High'].shift(-1) - df['High'].shift(-2))/df['High'].shift(-2)
    df['Feature_3'] = (df['Low'].shift(-1) - df['Low'].shift(-2)) / df['Low'].shift(-2)
    df['Feature_4'] = (df['Volume'] - df['Volume'].shift(-1))/df['Volume'].shift(-1)
    df['Label'] = (df['Close'] - df['Close'].shift(-1))
    df.fillna(method='pad', inplace=True)
    df['Label'] = np.sign(df['Label']).astype(int)
    return df


def join_data():
    # Join data into one csv file
    with open("sp25tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    result = pd.DataFrame()
    for ticker in tickers:
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker), index_col=0)
        if result.empty:
            result = df
        else:
            result = result.append(df)
    result.to_csv('sp25_joined.csv')
    return result


def visualize_data(ticker_list=["AMZN","BAC"]):
    for ticker in ticker_list:
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker), index_col=0)
        df['Open'].plot(title="Open price for {}".format(ticker))
        plt.xticks(rotation=30, size=6)
        plt.show()


def main():
    get_data(True)
    join_data()
    result = fe.add_feature_industry()
    model_all.cross_validation(result)


if __name__ == "__main__":
    main()





