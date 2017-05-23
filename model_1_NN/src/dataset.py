import arff
import numpy as np

class DataSet(object):
    def __init__(self, config):
        self.config = config
        self.train, self.test, self.validation = None, None, None
        self.path = self.config.dataset_path

    def get_features(self, path):
        file_content = arff.load(open(path, "r"))
        data = list()
        for j in file_content['data']:
            small_data = list()
            for i in j:
                small_data.append(float(i))
            data.append(small_data)
        data = np.asarray(data)
        return data[:, : self.config.features_dim]

    def get_labels(self, path):
        file_content = arff.load(open(path, "r"))
        data = list()
        for j in file_content['data']:
            small_data = list()
            for i in j:
                small_data.append(float(i))
            data.append(small_data)
        data = np.asarray(data)
        return data[: , self.config.features_dim : self.config.features_dim + self.config.labels_dim]

    def get_train(self):
        if self.train == None:
            X = self.get_features(self.config.train_path)
            Y = self.get_labels(self.config.train_path)
            length = X.shape[0]
            X, Y = X[0 : int(0.8 * length) , :], Y[0 : int(0.8 * length), :]
            self.train = X, Y
        else :
            X, Y = self.train
        np.random.shuffle(X)
        np.random.shuffle(Y)
        print("=> Training-Set Generated")
        return X, Y

    def get_validation(self):
        if self.validation == None:
            X = self.get_features(self.config.train_path)
            Y = self.get_labels(self.config.train_path)
            length = X.shape[0]
            X, Y = X[0 : int(0.2 * length) , :], Y[0 : int(0.2 * length), :]
            self.validation = X, Y
        else :
            X, Y = self.validation
        np.random.shuffle(X)
        np.random.shuffle(Y)
        print("=> Validation-Set Generated")
        return X, Y

    def get_test(self):
        if self.test == None:
            X = self.get_features(self.config.train_path)
            Y = self.get_labels(self.config.train_path)
            self.test = X, Y
        else :
            X, Y = self.test
        np.random.shuffle(X)
        np.random.shuffle(Y)
        print("=> Test-Set Generated")
        return X, Y

    def next_batch(self, data):
        if data.lower() not in ["train", "test", "validation"]:
            raise ValueError
        func = {"train" : self.get_train, "test": self.get_test, "validation": self.get_validation}[data.lower()]
        X, Y = func()
        start = 0
        batch_size = self.config.batch_size
        total = len(X)
        while start < total :
            end = min(start + batch_size, total)
            x = X[start : end, :]
            y = Y[start : end, :]
            start += 1
            yield (x, y, int(total))