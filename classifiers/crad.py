from sklearn import tree
import pydotplus

from utils.data_getter import Data


def calculate(train: Data, test: Data, max_depth=2, write=True):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(train.data, train.target)

    if write:
        __write(clf, train)

    return clf.predict(test.data)


def __write(clf, data: Data):
    dot_data = tree.export_graphviz(clf, feature_names=data.feature_names, class_names=data.target_names,
                                    out_file=None, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("./output/voice.png")
