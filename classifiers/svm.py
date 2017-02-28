import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn import metrics

from utils.data_getter import Data


"""
conclusions
===========

linear kernel: c = 0.1
rbf kernel: gamma = 0.01
polynomial kernel: degree = 3
"""


def calculate(train: Data, test: Data):
    clf = SVC(kernel='linear', C=0.1)
    clf.fit(train.data, train.target)

    return clf.predict(test.data)


def get_data():
    df = pd.read_csv('../data/voice.csv')
    print("Total number of labels: {}".format(df.shape[0]))
    print("Number of male: {}".format(df[df.label == 'male'].shape[0]))
    print("Number of female: {}".format(df[df.label == 'female'].shape[0]))

    xs = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Encode label category
    # male -> 1, female -> 0
    gender_encoder = LabelEncoder()
    y = gender_encoder.fit_transform(y)

    # Scale the data to be between -1 and 1
    scaler = StandardScaler()
    scaler.fit(xs)
    xs = scaler.transform(xs)
    return xs, y


def default_kernels(xs_train, xs_test, y_train, y_test):
    svc = SVC()  # Default hyperparameters
    svc.fit(xs_train, y_train)
    y_pred = svc.predict(xs_test)
    print('Accuracy Score:')
    print(metrics.accuracy_score(y_test, y_pred))

    svc = SVC(kernel='linear')
    svc.fit(xs_train, y_train)
    y_pred = svc.predict(xs_test)
    print('Accuracy Score:')
    print(metrics.accuracy_score(y_test, y_pred))

    svc = SVC(kernel='rbf')
    svc.fit(xs_train, y_train)
    y_pred = svc.predict(xs_test)
    print('Accuracy Score:')
    print(metrics.accuracy_score(y_test, y_pred))

    svc = SVC(kernel='poly')
    svc.fit(xs_train, y_train)
    y_pred = svc.predict(xs_test)
    print('Accuracy Score:')
    print(metrics.accuracy_score(y_test, y_pred))


def cv_kernels(xs, y):
    svc = SVC(kernel='rbf')
    scores = cross_val_score(svc, xs, y, cv=10, scoring='accuracy')  # cv is cross validation
    print(scores)
    print(scores.mean())

    svc = SVC(kernel='poly')
    scores = cross_val_score(svc, xs, y, cv=10, scoring='accuracy')  # cv is cross validation
    print(scores)
    print(scores.mean())


def check_out_c(xs, y, c_range):
    acc_score = []
    for c in c_range:
        svc = SVC(kernel='linear', C=c)
        scores = cross_val_score(svc, xs, y, cv=10, scoring='accuracy')
        acc_score.append(scores.mean())
        print(f"> c: {c:0.4f}, mean score: {acc_score[-1]}")
    print(acc_score)

    # plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
    plt.plot(c_range, acc_score)
    plt.xlabel('Value of C for SVC')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()


def check_out_gamma(xs, y, gamma_range):
    acc_score = []
    for g in gamma_range:
        svc = SVC(kernel='rbf', gamma=g)
        scores = cross_val_score(svc, xs, y, cv=10, scoring='accuracy')
        acc_score.append(scores.mean())
        print(f"> gamma: {g:0.4f}, mean score: {acc_score[-1]}")
    print(acc_score)

    plt.plot(gamma_range, acc_score)
    plt.xlabel('Value of gamma for SVC ')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()


def check_out_poly(xs, y):
    degree = [2, 3, 4, 5, 6]
    acc_score = []
    for d in degree:
        svc = SVC(kernel='poly', degree=d)
        scores = cross_val_score(svc, xs, y, cv=10, scoring='accuracy')
        acc_score.append(scores.mean())
        print(f"> degree: {d}, mean score: {acc_score[-1]}")
    print(acc_score)

    plt.plot(degree, acc_score, color='r')
    plt.xlabel('degrees for SVC ')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()


def main():
    xs, y = get_data()

    xs_train, xs_test, y_train, y_test = train_test_split(xs, y, test_size=0.2, random_state=1)
    default_kernels(xs_train, xs_test, y_train, y_test)

    cv_kernels(xs, y)

    check_out_c(xs, y, list(np.arange(1, 16)))
    check_out_c(xs, y, list(np.arange(0.1, 7, 0.1)))

    check_out_gamma(xs, y, [0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
    check_out_gamma(xs, y, [0.0001, 0.001, 0.01, 0.1])
    check_out_gamma(xs, y, [0.01, 0.02, 0.03, 0.04, 0.05])

    check_out_poly(xs, y)


if __name__ == '__main__':
    main()
