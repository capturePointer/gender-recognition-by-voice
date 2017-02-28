import csv
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def load_data(file_name):
    data = Data()
    data.load_from_csv(file_name, pandas=True)
    return data


class Data:
    def __init__(self):
        self.__data = []
        self.__target = []
        self.__feature_names = []
        self.__target_names = []

    def __len__(self):
        return len(self.__data)

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        self.__data = list(data)

    @property
    def target(self):
        return self.__target

    @target.setter
    def target(self, target):
        self.__target = list(target)

    @property
    def feature_names(self):
        return self.__feature_names

    @feature_names.setter
    def feature_names(self, feature_names):
        self.__feature_names = list(feature_names)

    @property
    def target_names(self):
        return self.__target_names

    @target_names.setter
    def target_names(self, target_names):
        self.__target_names = list(target_names)

    def load_from_csv(self, file_name, pandas=False):
        if pandas:
            self.__load_with_pandas(file_name)
        else:
            self.__load_from_csv(file_name)

    def __load_from_csv(self, file_name):
        with open(file_name, 'r') as file:
            reader = csv.DictReader(file)
            self.__feature_names = reader.fieldnames[:-1]
            for row in reader:
                r = [value for value in row.values()]
                data, label = list(map(float, r[:-1])), self.__add_target(r[-1])
                self.__data.append(data)
                self.__target.append(label)

    def __load_with_pandas(self, file_name):
        df = pd.read_csv(file_name)
        xs = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        scaler = StandardScaler()
        scaler.fit(xs)
        self.__data = scaler.transform(xs)

        encoder = LabelEncoder()
        encoder.fit(y)
        self.__target = encoder.transform(y)
        self.__target_names = encoder.classes_
        self.__feature_names = df.columns

    def split(self, test_size, random_state=1):
        train, test = Data(), Data()
        train.feature_names = self.feature_names
        test.feature_names = self.feature_names

        train.target_names = self.target_names
        test.target_names = self.target_names

        train.data, test.data, train.target, test.target = train_test_split(self.__data, self.__target,
                                                                            test_size=test_size,
                                                                            random_state=random_state)
        return train, test

    def __add_target(self, target):
        if target in self.__target_names:
            return self.__target_names.index(target)
        else:
            self.__target_names.append(target)
            return len(self.__target_names) - 1
