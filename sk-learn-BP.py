from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import random


class DataSet(object):
    def __init__(self, filename="./flower.csv"):
        self.data = []
        self.train_data = []
        self.train_tag = []
        self.test_data = []
        self.test_tag = []
        self.filename = filename
        self.read_csv()

    def read_csv(self):
        with open(self.filename, 'r') as f:
            data = f.readlines()
        for line in data:
            line = line.strip().split(',')
            tmp = []
            for ex in line:
                tmp.append(float(ex))
            self.data.append(tmp)

    def split_data(self, ratio):
        copy = self.data
        len_train_data = int(ratio * len(self.data))
        while len(self.train_data) < len_train_data:
            index = random.randint(0, len(copy) - 1)
            tmp = copy.pop(index)
            self.train_data.append(tmp[:-1])
            self.train_tag.append(self._tans_tag(tmp[-1]))
        for line in copy:
            self.test_data.append(line[:-1])
            self.test_tag.append(self._tans_tag(line[-1]))

    @staticmethod
    def _tans_tag(x):
        if x == 0.0:
            return 1
        elif x == 0.5:
            return 2
        else:
            return 3


if __name__ == '__main__':
    data_set = DataSet()
    data_set.split_data(0.7)
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32))
    clf.fit(data_set.train_data, data_set.train_tag)
    pre_tag = clf.predict(data_set.test_data).tolist()
    print("recall_score:", metrics.recall_score(data_set.test_tag, pre_tag, average='micro'))
    print("accuracy_score", metrics.accuracy_score(data_set.test_tag, pre_tag))