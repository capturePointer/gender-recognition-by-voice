from sklearn import linear_model

from utils.data_getter import Data


def calculate(train: Data, test: Data, show_coef=True, sort_coef=True):
    logistic = linear_model.LogisticRegression()
    logistic.fit(train.data, train.target)

    if show_coef:
        show(logistic, train.feature_names[:-1], sort_coef)

    return logistic.predict(test.data)


def show(logistic, features, sort):
    print("Coefficients:")
    coefs = list(zip(features, logistic.coef_[0]))
    if sort:
        coefs.sort(key=lambda x: -abs(x[1]))
    for feature, coef in coefs:
        print(f" > {feature:8s} {coef:+0.4f}")
