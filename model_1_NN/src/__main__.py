import os
import time
import utils
import numpy as np
import tensorflow as tf
from parser import Parser
from config import Config
from network import Network
from dataset import DataSet
from eval_performance import evaluate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Model(object):
    def __init__(self, config):
        self.epoch_count = 0
        self.config = config
        self.data = DataSet(config)
        self.add_placeholders()
        self.net = Network(config, self.x, self.keep_prob)
        self.optimizer = self.config.solver.optimizer
        self.y_pred = self.net.prediction()
        self.loss = self.net.loss(self.y_pred, self.y)
        self.train = self.net.train_step(self.loss)
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()

    def add_placeholders(self):
        self.x = tf.placeholder(tf.float32, shape = [None, self.config.features_dim])
        self.y = tf.placeholder(tf.float32, shape = [None, self.config.labels_dim])
        self.keep_prob = tf.placeholder(tf.float32)

    def run_epoch(self, sess, data):
        err = list()
        i = 0
        self.epoch_count += 1
        for X, Y, tot in self.data.next_batch(data):
            if i > tot :
                i = 0
                break
            feed_dict = {self.x : X, self.y : Y, self.keep_prob : self.config.solver.dropout}
            _, loss, Y_pred = sess.run([self.train, self.loss, self.y_pred], feed_dict = feed_dict)
            err.append(loss) 
            output = "Epoch ({}), Run ({}/{}) : Loss = {}".format(self.epoch_count, i, tot, loss)
            with open("../stdout/{}_train.log".format(self.config.project_name), "a+") as log:
                log.write(output + "\n")
            print("   {}".format(output), end='\r')
            i += 1
        return np.mean(err)

    def run_eval(self, sess, data):
        y, y_pred = list(), list()
        for X, Y, tot in self.data.next_batch(data):
            feed_dict = {self.x : X, self.y : Y, self.keep_prob : 1}
            loss, Y_pred = sess.run([self.loss, self.y_pred], feed_dict = feed_dict)
            y.append(Y)
            y_pred.append(Y_pred)
        metrics = evaluate(predictions = np.array(y_pred), labels = np.array(y))
        return loss, metrics
    ''' 
     TODO
     def add_summaries(self, sess):
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer_train = tf.summary.FileWriter(self.config.logs_dir + "train", sess.graph)
        summary_writer_val = tf.summary.FileWriter(self.config.logs_dir + "validation", sess.graph)
        summary_writer_test = tf.summary.FileWriter(self.config.logs_dir + "test", sess.graph)
        summary_writers = {'train': summary_writer_train, 'val': summary_writer_val, 'test': summary_writer_test}
        return summary_writers
    '''

    def fit(self, sess):
        '''
         - Patience Method : 
         + Train for particular no. of epochs, and based on the frequency, evaluate the model using validation data.
         + If Validation Loss increases, decrease the patience counter.
         + If patience becomes less than a certain threshold, devide learning rate by 10 and switch back to old model
         + If learning rate is lesser than a certain 
        '''
        max_epochs = self.config.max_epochs
        patience = self.config.patience
        patience_increase = self.config.patience_increase
        improvement_threshold = self.config.improvement_threshold
        best_validation_loss = 1e6

        step, best_step, losses, learning_rate = 1, -1, list(), self.config.solver.learning_rate
        while step <= max_epochs :
            start_time = time.time()
            average_loss = self.run_epoch(sess, "train")
            duration = time.time() - start_time

            if step % self.config.epoch_freq == 0 :
                val_loss, metrics = self.run_eval(sess, "validation")
                print("--- Epoch : average_loss = {}".format(average_loss))
                if val_loss < best_validation_loss :
                    if val_loss < best_validation_loss * improvement_threshold :
                        self.saver.save(sess, self.config.ckptdir_path + "model_best.ckpt")
                        best_validation_loss = val_loss
                        best_step = step
                else :
                    if patience < 1:
                        self.saver.restore(sess, self.config.ckptdir_path + "model_best.ckpt")
                        if learning_rate <= 0.00001 :
                            print("=> Breaking by Patience Method")
                            break
                        else :
                            learning_rate /= 10
                            patience = self.config.patience
                            print("=> Learning rate dropped to {}".format(learning_rate))
                    else :
                        patience -= 1
        print("=> Best epoch : {}".format(best_step))
        test_loss, test_metrics = self.run_epoch(sess, "train")
        return {"train" : best_validation_loss, "test" : test_loss}

def init_model(config):
    tf.reset_default_graph()
    tf.set_random_seed(1234)
    with tf.variable_scope('Model', reuse=None) as scope:
        model = Model(config)

    tf_config = tf.ConfigProto(allow_soft_placement=True)#, device_count = {'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    sm = tf.train.SessionManager()

    if config.retrain:
        print("=> Loading model from checkpoint")
        load_ckpt_dir = config.ckptdir_path
    else:
        print("=> No model loaded from checkpoint")
        load_ckpt_dir = ''
    sess = sm.prepare_session("", init_op = model.init, saver = model.saver, checkpoint_dir = load_ckpt_dir, config = tf_config)
    return model, sess

def train_model(config):
    print("\033[92m=>\033[0m Training Model")
    model, sess = init_model(config)
    with sess:
        #summary_writers = model.add_summaries(sess)
        loss_dict = model.fit(sess)
        return loss_dict

def main():
    args = Parser().get_parser().parse_args()
    config = Config(args)
    train_model(config)

if __name__ == '__main__' :
    np.random.seed(1234)
    main() # Phew!