import utils
import numpy as np
import tensorflow as tf
from config import Config

class Network(object):

    def __init__(self, config, summarizer):
        tf.set_random_seed(1234)
        self.summarizer = summarizer
        self.config = config
        self.Wx1, self.Wx2, self.Wx3, self.bx1, self.bx2, self.bx3 = self.init_Fx_variables()
        self.We1, self.We2, self.be1, self.be2 = self.init_Fe_variables()
        self.Wd1, self.Wd2, self.bd1, self.bd2 = self.init_Fd_variables()

    def weight_variable(self, shape, name):
        return tf.get_variable(name=name, shape=shape,  initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name):
        return tf.Variable(tf.constant(0.1, shape=shape), name=name)

    def init_Fx_variables(self):
        W1 = self.weight_variable([self.config.features_dim, self.config.solver.hidden_dim], "weight_x1")
        W2 = self.weight_variable([self.config.solver.hidden_dim, self.config.solver.hidden_dim], "weight_x2")
        W3 = self.weight_variable([self.config.solver.hidden_dim, self.config.solver.latent_embedding_dim], "weight_x3")
        b1 = self.bias_variable([self.config.solver.hidden_dim], "bias_x1")
        b2 = self.bias_variable([self.config.solver.hidden_dim], "bias_x2")
        b3 = self.bias_variable([self.config.solver.latent_embedding_dim], "bias_x3")
        return W1, W2, W3, b1, b2, b3

    def init_Fe_variables(self):
        W1 = self.weight_variable([self.config.labels_dim, self.config.solver.hidden_dim], "weight_e1")
        W2 = self.weight_variable([self.config.solver.hidden_dim, self.config.solver.latent_embedding_dim], "weight_e2")
        b1 = self.bias_variable([self.config.solver.hidden_dim], "bias_e1")
        b2 = self.bias_variable([self.config.solver.latent_embedding_dim], "bias_e2") 
        return W1, W2, b1, b2

    def init_Fd_variables(self):
        W1 = self.weight_variable([self.config.solver.latent_embedding_dim, self.config.solver.hidden_dim], "weight_d1")
        W2 = self.weight_variable([self.config.solver.hidden_dim, self.config.labels_dim], "weight_d2")
        b1 = self.bias_variable([self.config.solver.hidden_dim], "bias_d1")
        b2 = self.bias_variable([self.config.labels_dim], "bias_d2")
        return W1, W2, b1, b2

    def accuracy(self, y_pred, y):
        return tf.reduce_mean(tf.cast(tf.equal(tf.round(y_pred), y), tf.float32))

    def Fx(self, X, keep_prob):
        hidden1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, self.Wx1) + self.bx1), keep_prob)
        hidden2 = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden1, self.Wx2) + self.bx2), keep_prob)
        hidden3 = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden2, self.Wx3) + self.bx3), keep_prob)
        return hidden3

    def Fe(self, Y, keep_prob):
        hidden1 = tf.nn.dropout(tf.nn.relu(tf.matmul(Y, self.We1)) + self.be1, keep_prob)
        pred = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden1, self.We2) + self.be2), keep_prob)
        return pred

    def Fd(self, input, keep_prob):
        hidden1 = tf.nn.dropout(tf.nn.relu(tf.matmul(input, self.Wd1) + self.bd1), keep_prob)
        y_pred = tf.matmul(hidden1, self.Wd2) + self.bd2
        return y_pred

    def prediction(self, X, keep_prob):
        Fx = self.Fx(X, keep_prob)
        return self.Fd(Fx, keep_prob)

    def embedding_loss(self, features, labels, keep_prob):
        Fx = self.Fx(features, keep_prob)
        Fe = self.Fe(labels, keep_prob)
        return tf.reduce_sum(tf.square(Fx - Fe))

    '''
    def output_loss(self, X, Y, keep_prob):
        Ei = 0.0
        prediction = 1.0 / (1 + np.exp(-self.prediction(X, keep_prob)))
        for i in xrange(Y.shape[0] - 1):
            prediction_, Y_ = prediction[i : i+1, :], Y[i : i+1, :]
            mask = np.logical_and(Y_, np.logical_not(Y_.T)).astype(float)
            y1 = np.sum(Y_)
            y0 = Y_.shape[1] - Y1
            temp = (1/y1 * y0) * np.sum(np.exp(-np.multiply(mask, prediction_ - prediction_.T)))
            if(not np.isinf(temp) or not np.isnan(temp)):
                Ei += temp
        return Ei

    def output_loss(self, X, Y, keep_prob):
        Ei = 0.0
        prediction = tf.nn.sigmoid(self.prediction(X, keep_prob))
        for i in range(self.config.batch_size - 1):
            prediction_ = tf.slice(prediction, [i, 0], [1, self.config.labels_dim])
            Y_          = tf.slice(Y,          [i, 0], [1, self.config.labels_dim])
            mask = tf.cast(tf.logical_and(tf.cast(Y_[0], tf.bool), \
                   tf.logical_not(tf.cast(tf.transpose(Y_[0]), tf.bool))), tf.float32)
            y1 = tf.reduce_sum(Y_[0])
            y0 = self.config.labels_dim - y1
            temp = (1/(y1 * y0)) * (tf.reduce_sum(tf.exp(-tf.multiply(mask, prediction_[0] - tf.transpose(prediction_[0])))))
            temp = tf.Print(temp, [temp, tf.reduce_sum(mask)], message='-----------temp: ')
            Ei += tf.cond(tf.logical_or(tf.is_inf(temp), tf.is_nan(temp)), lambda : tf.constant(0.0), lambda : temp)
        return Ei 
    '''

    def cross_loss(self, features, labels, keep_prob):
        predictions = self.prediction(features, keep_prob)
        cross_loss = tf.add(tf.log(1e-10 + tf.nn.sigmoid(predictions)) * labels, tf.log(1e-10 + (1 - tf.nn.sigmoid(predictions))) * (1 - labels))
        cross_entropy_label = -1 * tf.reduce_mean(tf.reduce_sum(cross_loss, 1))
        return cross_entropy_label

    def loss(self, features, labels, keep_prob):
        return self.cross_loss(features, labels, keep_prob) # tf.py_func(self.output_loss, [features, labels, keep_prob], tf.float32) * self.config.solver.alpha + self.embedding_loss(features, labels, keep_prob) #

    def train_step(self, loss):
        optimizer = self.config.solver.optimizer
        return optimizer(self.config.solver.learning_rate).minimize(loss)