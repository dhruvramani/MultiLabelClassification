import os
import sys
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
        self.summarizer = tf.summary
        self.net = Network(config, self.summarizer)
        self.optimizer = self.config.solver.optimizer
        self.y_pred = self.net.prediction(self.x, self.keep_prob)
        self.predict = self.net.predict(self.x)
        self.loss = self.net.loss(self.y_pred, self.y)
        self.accuracy = self.net.accuracy(self.predict, self.y)
        self.summarizer.scalar("loss", self.loss)
        self.summarizer.scalar("accuracy", self.accuracy)
        self.train = self.net.train_step(self.loss)
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()

    def add_placeholders(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.config.features_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.config.labels_dim])
        self.keep_prob = tf.placeholder(tf.float32)

    def run_epoch(self, sess, data):
        err = list()
        i = 0
        self.epoch_count += 1
        if self.config.load or self.config.debug:
            path_ = "../results/tensorboard"
        else :
            path_ = "../bin/results/tensorboard"
        writer = self.summarizer.FileWriter(path_)
        merged_summary = self.summarizer.merge_all()
        for X, Y, tot in self.data.next_batch(data):
            if i >= tot :
                break
            feed_dict = {self.x : X, self.y : Y, self.keep_prob : self.config.solver.dropout}
            if not self.config.load:
                _, loss, Y_pred = sess.run([self.train, self.loss, self.predict], feed_dict=feed_dict)
                err.append(loss) 
                output = "Epoch ({}) Batch({}) : Loss = {}".format(self.epoch_count, i // self.config.batch_size , loss)
                with open("../stdout/{}_train.log".format(self.config.project_name), "a+") as log:
                    log.write(output + "\n")
                print("   {}".format(output), end='\r')
            summ = sess.run(merged_summary, feed_dict={self.x : X, self.y: Y, self.keep_prob : 1})
            writer.add_summary(summ)
            i += self.config.batch_size
        return np.mean(err)

    def run_eval(self, sess, data):
        y, y_pred, loss_, metrics = list(), list(), 0.0, None
        accuracy = 0.0
        next_batch = self.data.next_batch(data)
        for X, Y, tot in next_batch:
            feed_dict = {self.x : X, self.y : Y, self.keep_prob : 1}
            loss_, Y_pred, accuracy_val = sess.run([self.loss, self.predict, self.accuracy], feed_dict=feed_dict)
            metrics = evaluate(predictions=np.array(Y_pred), labels=np.array(Y))
            accuracy += accuracy_val #metrics['accuracy']
        return loss_ / self.config.batch_size, accuracy / self.config.batch_size, metrics 
    
    def fit(self, sess):
        '''
         - Patience Method : 
         + Train for particular no. of epochs, and based on the frequency, evaluate the model using validation data.
         + If Validation Loss increases, decrease the patience counter.
         + If patience becomes less than a certain threshold, devide learning rate by 10 and switch back to old model
         + If learning rate is lesser than a certain 
        '''
        max_epochs = self.config.max_epochs
        if self.config.debug == True:
            max_epochs = 5
        patience = self.config.patience
        patience_increase = self.config.patience_increase
        improvement_threshold = self.config.improvement_threshold
        best_validation_loss = 1e6
        self.epoch_count = 0
        step, best_step, losses, learning_rate = self.epoch_count, -1, list(), self.config.solver.learning_rate
        while step <= max_epochs :
            if(self.config.load == True):
                break
            start_time = time.time()
            average_loss = self.run_epoch(sess, "train")
            duration = time.time() - start_time
            if not self.config.debug :
                if step % self.config.epoch_freq == 0 :
                    val_loss, accuracy, metrics = self.run_eval(sess, "validation")
                    output =  "=> Training : \naverage_loss = {} | Validation : Accuracy = {}".format(average_loss, accuracy)
                    output += "\n=> Validation : \nCoverage = {}, Average Precision = {}\n Micro Precision = {}, Micro Recall = {}, Micro F Score = {}".format(metrics['coverage'], metrics['average_precision'], metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'])
                    output += "\n Macro Precision = {}, Macro Recall = {}, Macro F Score = {}".format(metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'])
                    with open("../stdout/validation.log", "a+") as f:
                        f.write(output)
                    print(output)
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
            step = self.epoch_count
        print("=> Best epoch : {}".format(best_step))
        if self.config.debug == True:
            sys.exit()
        test_loss, test_accuracy, test_metrics = self.run_eval(sess, "test")
        returnDict = {"test_loss" : test_loss, "test_accuracy" : test_accuracy, 'test_metrics' : test_metrics}
        if self.config.debug == False:
            returnDict["train"] =  best_validation_loss
        return returnDict


def init_model(config):
    tf.reset_default_graph()
    tf.set_random_seed(1234)
    with tf.variable_scope('Model', reuse=None) as scope:
        model = Model(config)

    tf_config = tf.ConfigProto(allow_soft_placement=True)#, device_count = {'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    sm = tf.train.SessionManager()

    if config.retrain or config.load == True:
        print("=> Loading model from checkpoint")
        load_ckpt_dir = config.ckptdir_path
    else:
        print("=> No model loaded from checkpoint")
        load_ckpt_dir = ''
    sess = sm.prepare_session("", init_op=model.init, saver=model.saver, checkpoint_dir=load_ckpt_dir, config=tf_config)
    if config.load == True :
        saver = tf.train.Saver()
        sess_ = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        saver.restore(sess_, config.ckptdir_path + "/resultsmodel_best.ckpt")
        return model, sess_
    return model, sess

def train_model(config):
    if config.load == True:
        print("\033[92m=>\033[0m Testing Model")
    else: 
        print("\033[92m=>\033[0m Training Model")
    model, sess = init_model(config)
    with sess:
        #summary_writers = model.add_summaries(sess)
        loss_dict = model.fit(sess)
        return loss_dict

def main():
    args = Parser().get_parser().parse_args()
    config = Config(args)
    loss_dict = train_model(config)
    metrics = loss_dict['test_metrics']
    if config.debug == False:
        output = "=> Best Train Loss : {}, Test Loss : {}, Test Accuracy : {}".format(loss_dict["train"], loss_dict["test_loss"], loss_dict["test_accuracy"])
    else : 
        output = "=> Test Loss : {}, Test Accuracy : {}".format(loss_dict["test_loss"], loss_dict["test_accuracy"])
    output += "\n=> Test : Coverage = {}, Average Precision = {}, Micro Precision = {}, Micro Recall = {}, Micro F Score = {}".format(metrics['coverage'], metrics['average_precision'], metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'])
    output += "\n=> Test : Macro Precision = {}, Macro Recall = {}, Macro F Score = {}".format(metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'])
    with open("../stdout/test_log.log", "a+") as f:
        f.write(output)
    print("\033[1m\033[92m{}\033[0m\033[0m".format(output))

if __name__ == '__main__' :
    np.random.seed(1234)
    main() # Phew!