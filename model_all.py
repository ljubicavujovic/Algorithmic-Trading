from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
import feature_extension as fe
import intro
import datetime as dt
style.use('ggplot')


def get_mask(X, time):
    mask = np.ones(len(X), dtype=bool)
    for i, row in enumerate(X.itertuples()):
        if dt.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S") <= time:
            mask[i] = True
        else:
            mask[i] = False
    return mask


def get_mask_backup(X, time):
    mask = np.ones(len(X), dtype=bool)
    for i, row in enumerate(X.itertuples()):
        if int(row[0][-9:-6]) <= time:
            mask[i] = True
        else:
            mask[i] = False
    return mask


def prepare_data(result, time):
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
    return train_X, train_y, test_X, test_y


def make_model_svm(result, time):
    train_X, train_y, test_X, test_y = prepare_data(result, time)
    clf = svm.SVC(C=0.8)
    clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)
    train_precision = sum(clf.predict(train_X) == train_y) * 1.0 / len(train_y)
    test_precision = sum(test_y == predictions) * 1.0 / len(test_y)
    return train_precision, test_precision


def make_model_random_forest(result, time):

    train_X, train_y, test_X, test_y = prepare_data(result, time)
    clf = RandomForestClassifier(n_estimators=70)
    clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)
    train_precision = sum(clf.predict(train_X) == train_y) * 1.0 / len(train_y)
    test_precision = sum(test_y == predictions) * 1.0 / len(test_y)
    return train_precision, test_precision


def make_model_gradient_boosting(result, time):
    train_X, train_y, test_X, test_y = prepare_data(result, time)
    clf = GradientBoostingClassifier(n_estimators=250, learning_rate=0.1)
    clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)
    train_precision = sum(clf.predict(train_X) == train_y) * 1.0 / len(train_y)
    test_precision = sum(test_y == predictions) * 1.0 / len(test_y)
    return train_precision, test_precision


def make_model_ada_boosting(result, time):
    train_X, train_y, test_X, test_y = prepare_data(result, time)
    clf = AdaBoostClassifier(n_estimators=250, learning_rate=0.1)
    clf.fit(train_X, train_y)
    train_precision = clf.score(train_X, train_y)
    test_precision = clf.score(test_X, test_y)
    return train_precision, test_precision


def cross_validation(result, name, model):
    train = []
    test = []
    start = dt.datetime(2017, 4, 19, 17)
    times = [start + dt.timedelta(minutes=30) * i for i in range(0, 10)]
    timeticks = [t.strftime("%H:%M") for t in times]
    for t in times:
        train_precision, test_precision = model(result, t)
        train.append(train_precision)
        test.append(test_precision)
    plt.plot(times, train, 'r-', times, test, 'b-')
    plt.legend(['Train score', 'Test score'])
    plt.xlabel('Time')
    plt.ylabel('Precision of model')
    plt.title('Precision of train and test dataset depending \n on the time until trained - {}'.format(name), size=12)
    plt.xticks(times, timeticks, rotation=15, size=8)
    plt.savefig('Precision of train and test dataset depending on the time until trained - {}.png'.format(name))
    plt.close()
    return train, test


def feature_correlation(df, features=None):
    if features is not None:
        df = df[features]
    correlation_matrix = df.corr()

    data = correlation_matrix.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = correlation_matrix.columns
    row_labels = correlation_matrix.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()
    plt.savefig('Correlation Heatmap.png')
    plt.close()
    return correlation_matrix


def results():
    result = pd.read_csv('sp25_joined.csv', index_col=0)
    algorythms = {'SVM': make_model_svm, 'Gradient Boosting Classifier': make_model_gradient_boosting, 'Ada Boosting Classifier': make_model_ada_boosting}

    with open('results.txt','w') as f:
        for algorythm in algorythms:
            train, test = cross_validation(result, algorythm, algorythms[algorythm])
            f.write("Results for {} algorythm: \n".format(algorythm))
            f.write("Train precision: {} \n".format(" ".join(str(x) for x in train)))
            f.write("Test precision: {}\n".format(" ".join(str(x) for x in test)))


def testing_beta():
    periods = range(10, 40, 5)
    start = dt.datetime(2017, 4, 19, 17)
    times = [start + dt.timedelta(minutes=30) * i for i in range(0, 10)]
    timeticks = [t.strftime("%H:%M") for t in times]
    for period in periods:
        fe.add_beta(period)
        result = intro.join_data()
        train = []
        test = []
        for t in times:
            train_precision, test_precision = make_model_svm(result, t)
            train.append(train_precision)
            test.append(test_precision)
        plt.plot(times, train, 'r-', times, test, 'b-')
        plt.legend(['Train score', 'Test score'])
        plt.xlabel('Time')
        plt.ylabel('Precision of model')
        plt.title('Precision of train and test dataset depending \n on the time until trained for Beta period = {}'.format(period), size=12)
        plt.xticks(times, timeticks, rotation=15, size=8)
        plt.savefig('Precision of train and test dataset depending on the time until trained for Beta period = {}'.format(period))
        plt.close()

results()
