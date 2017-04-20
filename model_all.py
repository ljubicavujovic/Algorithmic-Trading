from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np


def get_mask(X, time):
    mask = np.ones(len(X), dtype=bool)
    for i, row in enumerate(X.itertuples()):
        if int(row[0][-9:-6]) <= time:
            mask[i] = True
        else:
            mask[i] = False
    return mask


def make_model(result, time):
    y = result['Label']
    y.fillna(method='bfill', inplace=True)
    X = result[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Industry_ID', 'Beta']]
    X.fillna(method='pad', inplace=True)
    X.fillna(method='bfill', inplace=True)
    mask = get_mask(X, time)
    train_X = X[mask]
    train_y = y[mask]
    test_X = X[~mask]
    test_y = y[~mask]
    clf = svm.SVC()
    clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)
    train_precision = sum(clf.predict(train_X) == train_y) * 1.0 / len(train_y)
    test_precision = sum(test_y == predictions) * 1.0 / len(test_y)
    return train_precision, test_precision


def cross_validation(result):
    train = []
    test = []
    time = range(17, 22)
    for t in time:
        train_precision, test_precision = make_model(result, t)
        train.append(train_precision)
        test.append(test_precision)
    plt.plot(time, train, 'r--', time, test, 'b--')
    plt.legend(['Train score', 'Test score'])
    plt.xlabel('Time')
    plt.ylabel('Precision of model')
    plt.title('Precision of train and test dataset depending on the time until trained')
    plt.show()
    return train, test


def feature_correlation(df, features=None):
    if features is not None:
        df = df[features]
    correlation_matrix = df.corr()
    return correlation_matrix

