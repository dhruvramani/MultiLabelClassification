import os
import arff
import numpy as np

def get_features(path, features_dim):
    file_content = arff.load(open(path, "r"))
    data = list()
    for j in file_content['data']:
        small_data = list()
        for i in j:
            small_data.append(float(i))
        data.append(small_data)
    data = np.asarray(data)
    return data[:, : features_dim]

def get_labels(path, features_dim, labels_dim):
    file_content = arff.load(open(path, "r"))
    data = list()
    for j in file_content['data']:
        small_data = list()
        for i in j:
            small_data.append(float(i))
        data.append(small_data)
    data = np.asarray(data)
    return data[: , features_dim : features_dim + labels_dim]

def set_dims(dataset_path):
    with open(os.path.join(dataset_path, "count.txt"), "r") as f:
        return [int (i) for i in f.read().split("\n") if i != ""]

if __name__ == '__main__':
    # Convert and save arff files to numpy-pickles for faster data I/O.
    print("Starting Training")
    features_dim, labels_dim = set_dims("./tmc2007/")
    train_features, train_labels = get_features("./tmc2007/tmc2007-train.arff", features_dim), get_labels("./tmc2007/tmc2007-train.arff", features_dim, labels_dim)
    train_features.dump("./tmc2007/tmc2007-train-features.pkl")
    train_labels.dump("./tmc2007/tmc2007-train-labels.pkl")
    print("End Train")
    print("Start Test")
    test_features, test_labels = get_features("./tmc2007/tmc2007-test.arff", features_dim), get_labels("./tmc2007/tmc2007-test.arff", features_dim, labels_dim)
    test_features.dump("./tmc2007/tmc2007-test-features.pkl")
    test_labels.dump("./tmc2007/tmc2007-test-labels.pkl")
