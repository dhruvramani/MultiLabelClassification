import utils
import numpy as np
import tensorflow as tf
from config import Config

class Network(object):

    def __init__(self, config, x, keep_prob):
        tf.set_random_seed(1234)
        self.config = config
        self.X, self.W1, self.W2, self.b1, self.b2 = self.init_variable(x, keep_prob)

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

    def init_variable(self, x, keep_prob):
        X  = tf.nn.dropout(x, keep_prob)
        W1 = self.weight_variable([self.config.features_dim, self.config.solver.hidden_dim], "weight_1")
        b1 = self.bias_variable([self.config.solver.hidden_dim], "bias_1")
        W2 = self.weight_variable([self.config.solver.hidden_dim, self.config.labels_dim], "weight_2")
        b2 = self.bias_variable([self.config.labels_dim], "bias_2")
        return X, W1, W2, b1, b2

    def prediction(self):
        hidden = tf.nn.relu(tf.matmul(self.X, self.W1) + self.b1)
        y_pred = tf.nn.sigmoid(tf.matmul(hidden, self.W2) + self.b2)
        return y_pred