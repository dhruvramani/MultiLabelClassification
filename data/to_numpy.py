import os
import arff
import numpy as np

def get_features(path, features_dim):
    file_content = arff.load(open(path, "r"))
    data=np.array(file_content['data'], dtype="float32")
    #for j in file_content['data']:
    #    small_data = list()
    #    for i in j:
    #        small_data.append(float(i))
    #    data.append(small_data)
    #data = np.asarray(data)
    return data[:, : features_dim]

def get_labels(path, features_dim, labels_dim):
    file_content = arff.load(open(path, "r"))
    data=np.array(file_content['data'], dtype="float32")
    #data = list()
    #for j in file_content['data']:
    #    small_data = list()
    #    for i in j:
    #        small_data.append(float(i))
    #    data.append(small_data)
    #data = np.asarray(data)
    return data[: , features_dim : ]

def set_dims(dataset_path):
    with open(os.path.join(dataset_path, "count.txt"), "r") as f:
        return [int (i) for i in f.read().split("\n") if i != ""]

if __name__ == '__main__':
    # Convert and save arff files to numpy-pickles for faster data I/O.
    print("Starting Training")
    features_dim, labels_dim = set_dims("./delicious/")
    train_features, train_labels = get_features("./delicious/delicious-train.arff", features_dim), get_labels("./delicious/delicious-train.arff", features_dim, labels_dim)
    train_features.dump("./delicious/delicious-train-features.pkl")
    train_labels.dump("./delicious/delicious-train-labels.pkl")
    print("End Train")
    print("Start Test")
    test_features, test_labels = get_features("./delicious/delicious-test.arff", features_dim), get_labels("./delicious/delicious-test.arff", features_dim, labels_dim)
    test_features.dump("./delicious/delicious-test-features.pkl")
    test_labels.dump("./delicious/delicious-test-labels.pkl")
