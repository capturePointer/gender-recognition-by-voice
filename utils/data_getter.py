import csv
from random import shuffle


def load_data(file_name):
    data = Data()
    data.load_form_csv(file_name)
    return data


class Data:
    def __init__(self):
        self.__data = []
        self.__target = []
        self.__feature_names = []
        self.__target_names = []

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

    def load_form_csv(self, file_name):
        with open(file_name, 'r') as file:
            reader = csv.DictReader(file)
            self.__feature_names = reader.fieldnames[:-1]
            for row in reader:
                r = [value for value in row.values()]
                data, label = list(map(float, r[:-1])), self.__add_target(r[-1])
                self.__data.append(data)
                self.__target.append(label)

    def split(self, ratio=0.8, random=False):
        limit = int(len(self.__data) * ratio)
        train, test = Data(), Data()

        train.feature_names = self.feature_names
        test.feature_names = self.feature_names

        train.target_names = self.target_names
        test.target_names = self.target_names

        if random:
            return self.__random_split(train, test, limit)

        return self.__straight_split(train, test, limit)

    def __straight_split(self, train, test, limit):
        train.data, test.data = self.__data[:limit], self.__data[limit:]
        train.target, test.target = self.__target[:limit], self.__target[limit:]
        return train, test

    def __random_split(self, train, test, limit):
        shuffled = list(zip(self.__data, self.__target))
        shuffle(shuffled)

        data = [chunk[0] for chunk in shuffled]
        target = [chunk[1] for chunk in shuffled]

        train.data, test.data = data[:limit], data[limit:]
        train.target, test.target = target[:limit], target[limit:]
        return train, test

    def __add_target(self, target):
        if target in self.__target_names:
            return self.__target_names.index(target)
        else:
            self.__target_names.append(target)
            return len(self.__target_names) - 1
