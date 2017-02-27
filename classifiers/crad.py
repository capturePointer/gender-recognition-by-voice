from sklearn import tree
import pydotplus


def do(train, test):
    clf = tree.DecisionTreeClassifier(max_depth=2)
    clf = clf.fit(train.data, train.target)

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=train.feature_names,
                                    class_names=train.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("./output/voice.png")
