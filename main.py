from utils.data_getter import load_data
from classifiers import crad, svm, logistic
from sklearn import metrics


def main():
    print('> Getting data...')
    voices = load_data('./data/voice.csv')
    train, test = voices.split(test_size=0.2, random_state=1)

    cs = [
        ('CRAD', lambda: crad.calculate(train, test, write=True, max_depth=None)),
        ('SVM', lambda: svm.calculate(train, test)),
        ('logistic regression', lambda: logistic.calculate(train, test)),
    ]

    print('> Calculating...')
    for name, func in cs:
        prediction = func()
        print(f"== {name} ==")
        print(f'accuracy score: {metrics.accuracy_score(test.target, prediction)}')
        print(f'f1-score: {metrics.f1_score(test.target, prediction)}')
        print()

if __name__ == '__main__':
    main()
