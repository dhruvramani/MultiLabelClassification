import numpy as np
from collections import Counter
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import coverage_error, label_ranking_loss, hamming_loss, accuracy_score

def patk(predictions, labels):
    pak = np.zeros(3)
    K = np.array([1, 3, 5], dtype=np.float32)
    for i in range(predictions.shape[0]):
        pos = np.argsort(-predictions[i, :])
        y = labels[i, :]
    y = y[pos]
    for j in range(3):
        k = K[j]
        pak[j] += (np.sum(y[:k]) / k)
        #print(type(pak[j]))
    pak = pak / predictions.shape[0]

    return pak

def cm_precision_recall(prediction, truth):
    """Evaluate confusion matrix, precision and recall for given set of labels and predictions
     Args
       prediction: a vector with predictions
       truth: a vector with class labels
     Returns:
       cm: confusion matrix
       precision: precision score
       recall: recall score"""
    confusion_matrix = Counter()

    positives = [1]

    binary_truth = [x in positives for x in truth]
    binary_prediction = [x in positives for x in prediction]

    for t, p in zip(binary_truth, binary_prediction):
        confusion_matrix[t, p] += 1

    cm = np.array([confusion_matrix[True, True], confusion_matrix[False, False], confusion_matrix[False, True],
                   confusion_matrix[True, False]])
    # print cm
    precision = (cm[0] / (cm[0] + cm[2] + 0.000001))
    recall = (cm[0] / (cm[0] + cm[3] + 0.000001))
    return cm, precision, recall

def evaluate(predictions, labels, threshold=0, multi_label=True):
    #predictions are logits here and binarized labels
    assert predictions.shape == labels.shape, "Shapes: %s, %s" % (predictions.shape, labels.shape,)
    metrics = dict()
    #metrics['cross_entropy'] = -np.mean(labels * np.log(predictions + 1e-8))

    if not multi_label:
        metrics['bae'] = BAE(labels, predictions)
        labels, predictions = np.argmax(labels, axis=1), np.argmax(predictions, axis=1)

        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'], _ = \
            precision_recall_fscore_support(labels, predictions, average='micro')
        metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'], metrics['coverage'], \
            metrics['average_precision'], metrics['ranking_loss'], metrics['pak'], metrics['hamming_loss'] \
            = 0, 0, 0, 0, 0, 0, 0, 0
    else:
        metrics['accuracy'] = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))
        if threshold:
            for i in range(predictions.shape[0]):
                predictions[i, :][predictions[i, :] >= threshold] = 1
                predictions[i, :][predictions[i, :] < threshold] = 0
        else: # TOP K
            for i in range(predictions.shape[0]):
                k = np.sum(labels[i])
                pos = predictions[i].argsort()
                predictions[i].fill(0)
                predictions[i][pos[-int(k):]] = 1

        #print("Predicted: ", predictions)
        #print("Truth: ", labels)

        metrics['bae'] = 0
        metrics['coverage'] = coverage_error(labels, predictions)
        metrics['average_precision'] = label_ranking_average_precision_score(labels, predictions)
        metrics['ranking_loss'] = label_ranking_loss(labels, predictions)
        #metrics['pak'] = patk(predictions, labels)
        metrics['hamming_loss'] = hamming_loss(labels, predictions)
        #metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'], metrics['macro_precision'], \
         #   metrics['macro_recall'], metrics['macro_f1'] = bipartition_scores(labels, predictions)

    return metrics