import utils
import numpy as np
import tensorflow as tf
from config import Config

class Network(object):

    def __init__(self, config):
        tf.set_random_seed(1234)
        self.config = config
        self.W1, self.W2, self.b1, self.b2 = self.init_variable()

    def weight_variable(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape = shape, stddev = 1.0 / shape[0]), name = name)

    def bias_variable(self, shape, name):
        return tf.Variable(tf.constant(0.1, shape = shape), name = name)

    def variable_summaries(self, var, name):
        with tf.name_sope('summaries'):
            tf.summary.histogram(name, var)

    def loss(self, y_pred, y):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_pred, labels = y))

    def train_step(self, loss):
        optimizer = self.config.solver.optimizer
        return optimizer(self.config.solver.learning_rate).minimize(loss)

    def init_variable(self):
        W1 = self.weight_variable([self.config.features_dim, self.config.solver.hidden_dim], "weight_1")
        b1 = self.bias_variable([self.config.solver.hidden_dim], "bias_1")
        W2 = self.weight_variable([self.config.solver.hidden_dim, self.config.labels_dim], "weight_2")
        b2 = self.bias_variable([self.config.labels_dim], "bias_2")
        return W1, W2, b1, b2

    def accuracy(self, y_pred, y):
        return tf.reduce_mean(tf.cast(tf.equal(tf.round(y_pred), y), tf.float32))

    def predict(self, X):
        hidden = tf.nn.relu(tf.matmul(X, self.W1) + self.b1)
        y_pred = tf.nn.sigmoid(tf.matmul(hidden, self.W2) + self.b2)
        return y_pred
    
    def prediction(self, X, keep_prob):
        hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(X, self.W1) + self.b1), keep_prob)
        y_pred = tf.matmul(hidden, self.W2) + self.b2
        return y_pred
