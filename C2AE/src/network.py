import utils
import numpy as np
import tensorflow as tf

class Network(object):

    def __init__(self, config, summarizer):
        tf.set_random_seed(1234)
        self.summarizer = summarizer
        self.config = config
        self.We1, self.We2, self.be1, self.be2 = self.init_Fe_variable()
        self.Wd1, self.Wd2, self.bd1, self.bd2 = self.init_Fd_variable()
        self.Wx1, self.Wx2, self.Wx3, self.bx1, self.bx2, self.bx3 = self.init_Fx_variable()

    def weight_variable(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name):
        # Most of them don't change
        return tf.Variable(tf.constant(0.1, shape=shape), name=name)

    def init_Fx_variable(self):
        Wx1 = self.weight_variable([self.config.features_dim, self.config.solver.hidden_dim], "weight_x_1")
        bx1 = self.bias_variable([self.config.solver.hidden_dim], "bias_x_1")
        Wx2 = self.weight_variable([self.config.solver.hidden_dim, self.config.solver.hidden_dim], "weight_x_2")
        bx2 = self.bias_variable([self.config.solver.hidden_dim], "bias_x_2")
        Wx3 = self.weight_variable([self.config.solver.hidden_dim, self.config.label_embedding_dim], "weight_x_3")
        bx3 = self.bias_variable([self.config.label_embedding_dim], "bias_x_3")
        '''
        self.summarizer.histogram('weight_x_1', Wx1)
        self.summarizer.histogram('weight_x_2', Wx2)
        self.summarizer.histogram('weight_x_3', Wx3)
        self.summarizer.histogram('bias_x_1', bx1)
        self.summarizer.histogram('bias_x_2', bx2)
        self.summarizer.histogram('bias_x_3', bx3)
        '''
        return Wx1, Wx2, Wx3, bx1, bx2, bx3

    def init_Fe_variable(self):
        We1 = self.weight_variable([self.config.labels_dim, self.config.solver.hidden_dim], "weight_e_1")
        We2 = self.weight_variable([self.config.solver.hidden_dim, self.config.label_embedding_dim], "weight_e_2")
        be1 = self.bias_variable([self.config.solver.hidden_dim], "bias_e_1")
        be2 = self.bias_variable([self.config.label_embedding_dim], "bias_e_2")
        '''
        self.summarizer.histogram('weight_e_1', We1)
        self.summarizer.histogram('weight_e_2', We2)
        self.summarizer.histogram('bias_e_1', be1)
        self.summarizer.histogram('bias_e_2', be2)
        '''
        return We1, We2, be1, be2

    def init_Fd_variable(self):
        Wd1 = self.weight_variable([self.config.label_embedding_dim, self.config.solver.hidden_dim], "weight_d_1")
        Wd2 = self.weight_variable([self.config.solver.hidden_dim, self.config.labels_dim], "weight_d_2")
        bd1 = self.bias_variable([self.config.solver.hidden_dim], "bias_d_1")
        bd2 = self.bias_variable([self.config.labels_dim], "bias_d_2")
        '''
        self.summarizer.histogram('weight_d_1', Wd1)
        self.summarizer.histogram('weight_d_2', Wd2)
        self.summarizer.histogram('bias_d_1', bd1)
        self.summarizer.histogram('bias_d_2', bd2)
        '''
        return Wd1, Wd2, bd1, bd2

    def leakyRelu(self, X, alpha=0.0001):
        return tf.maximum(alpha * X, X)

    def Fx(self, X):
        h1 = self.leakyRelu(tf.matmul(X, self.Wx1) + self.bx1)
        h2 = self.leakyRelu(tf.matmul(h1, self.Wx2) + self.bx2)
        return self.leakyRelu(tf.matmul(h2, self.Wx3) + self.bx3)

    def Fe(self, Y):
        h1 = self.leakyRelu(tf.matmul(Y, self.We1 + self.be1))
        return self.leakyRelu(tf.matmul(h1, self.We2) + self.be2)

    def Fd(self, X):        
        labels_embedding = self.Fx(X)
        h1 = self.leakyRelu(tf.matmul(labels_embedding, self.Wd1) + self.bd1)
        return tf.matmul(h1, self.Wd2) + self.bd2

    def predict(self, X):
        return tf.nn.sigmoid(self.Fd(X))

    def foo(self, X, Y):
        Fx, Fe = self.Fx(X), self.Fe(Y)
        return tf.reduce_mean(tf.square(Fx - Fe))

    def label_embedding_loss(self, X, Y, lagrange_const):
        Fx, Fe = self.Fx(X), self.Fe(Y)
        idenX, idenE = tf.constant(np.identity(self.config.batch_size), dtype=tf.float32), \
                       tf.constant(np.identity(self.config.batch_size), dtype=tf.float32)
        loss = tf.reduce_mean(tf.square(Fx - Fe)) + \
               lagrange_const * (tf.reduce_mean(tf.square(tf.matmul(Fx, tf.transpose(Fx)) - idenX)) + \
                                 tf.reduce_mean(tf.square(tf.matmul(Fe, tf.transpose(Fe)) - idenE)))
        return loss

    def output_loss(self, X, Y):
        Ei = 0.0
        prediction = self.predict(X)
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

    def loss(self, X, Y, lagrange_const, alpha):
        total_loss = self.label_embedding_loss(X, Y, lagrange_const) + alpha * self.output_loss(X, Y) 
        return total_loss #/ self.config.batch_size

    def train_step(self, loss):
        optimizer = self.config.solver.optimizer
        return optimizer(self.config.solver.learning_rate).minimize(loss)

    def accuracy(self, y_pred, y):
        return tf.reduce_mean(tf.cast(tf.equal(tf.round(y_pred), y), tf.float32))

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.float32)
    count = tf.reduce_sum(as_ints)
    return count
