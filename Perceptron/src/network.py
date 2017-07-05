import utils
import numpy as np
import tensorflow as tf
from config import Config

class Network(object):

    def __init__(self, config, summarizer):
        tf.set_random_seed(1234)
        self.summarizer = summarizer
        self.config = config
        #self.W1, self.W2, self.W3, self.b1, self.b2, self.b3 = self.init_variable()
        self.W1, self.W2, self.b1, self.b2 = self.init_variable()

    def weight_variable(self, shape, name):
        return tf.get_variable(name=name, shape=shape,  initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name):
        return tf.Variable(tf.constant(0.1, shape=shape), name=name)

    def loss(self, predictions, labels):
        lamda = 0.0
        cross_entropy = tf.add(tf.multiply(tf.log(1e-10 + tf.nn.sigmoid(predictions)), labels),
                        tf.multiply(tf.log(1e-10 + (1 - tf.nn.sigmoid(predictions))), (1 - labels)))
        loss = -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1), name='xentropy_mean')
        l2_losses = lamda * (tf.reduce_sum(tf.square(self.W1)) + tf.reduce_sum(tf.square(self.W2)))
        return l2_losses + loss # tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predictions)) 

    def train_step(self, loss):
        optimizer = self.config.solver.optimizer
        return optimizer(self.config.solver.learning_rate).minimize(loss)

    def init_variable(self):
        W1 = self.weight_variable([self.config.features_dim, self.config.solver.hidden_dim], "weight_1")
        W2 = self.weight_variable([self.config.solver.hidden_dim, self.config.labels_dim], "weight_2")
        b1 = self.bias_variable([self.config.solver.hidden_dim], "bias_1")
        b2 = self.bias_variable([self.config.labels_dim], "bias_2")
        self.summarizer.histogram('weight_1', W1)
        self.summarizer.histogram('weight_2', W2)
        self.summarizer.histogram('bias_1', b1)
        self.summarizer.histogram('bias_2', b2)
        return W1, W2, b1, b2

    def accuracy(self, y_pred, y):
        return tf.reduce_mean(tf.cast(tf.equal(tf.round(y_pred), y), tf.float32))
    
    def prediction(self, X, keep_prob):
        hidden1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, self.W1) + self.b1), keep_prob)
        y_pred = tf.matmul(hidden1, self.W2) + self.b2
        return y_pred