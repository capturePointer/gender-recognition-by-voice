from utils.data_getter import load_data
from classifiers import crad


def main():
    voices = load_data('./data/voice.csv')
    train, test = voices.split(ratio=0.5, random=True)

    crad.do(train, test)

if __name__ == '__main__':
    main()
