import arff
import numpy as np

class DataSet(object):
    def __init__(self, config):
        self.config = config
        self.start = 0
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
        X = self.get_features(self.config.train_path)
        Y = self.get_labels(self.config.train_path)
        np.random.shuffle(X)
        np.random.shuffle(Y)
        length = X.shape[0]
        print("=> Training-Set Generated")
        return X[0 : int(0.8 * length) , :], Y[0 : int(0.8 * length), :]

    def get_validation(self):
        X = self.get_features(self.config.train_path)
        Y = self.get_labels(self.config.train_path)
        np.random.shuffle(X)
        np.random.shuffle(Y)
        length = X.shape[0]
        print("=> Validation-Set Generated")
        return X[int(0.8 * length) : , :], Y[int(0.8 * length) : , :]

    def get_test(self):
        X = self.get_features(self.config.test_path)
        Y = self.get_labels(self.config.test_path)
        np.random.shuffle(X)
        np.random.shuffle(Y)
        print("=> Test-Set Generated")
        return X, Y

    def next_batch(self, data):
        if data.lower() not in ["train", "test", "validation"]:
            raise ValueError
        func = {"train" : self.get_train, "test": self.get_test, "validation": self.get_validation}[data.lower()]
        X, Y = func()
        batch_size = self.config.batch_size
        total = len(X)
        self.start = 0
        while self.start < batch_size :
            end = min(self.start + batch_size, total)
            x = X[self.start : end, :]
            y = Y[self.start : end, :]
            self.start += 1
            print(self.start)
            yield (x, y, total//batch_size)
