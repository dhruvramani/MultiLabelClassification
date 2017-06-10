import argparse

class Parser(object):

    def __init__(self):
        parser=argparse.ArgumentParser()
        parser.add_argument("--path", default="../", help="Base Path for the Folder")
        parser.add_argument("--project", default="C2AE", help="Project Folder")
        parser.add_argument("--folder_suffix", default="Default", help="Folder Name Suffix")
        parser.add_argument("--dataset", default="delicious", help="Name of the Dataset")
        parser.add_argument("--opt", default="adam", help="Optimizer : adam, rmsprop, sgd")
        parser.add_argument("--lr", default=0.01, help="Learning Rate", type=float)
        parser.add_argument("--hidden", default=512, help="Number of Units in the Hidden Layer", type=int)
        parser.add_argument("--batch_size", default=128, help="Batch Size", type=int)
        parser.add_argument("--dropout", default=0.5, help="Dropout Probab. for Pre-Final Layer", type=float)
        parser.add_argument("--lagrange", default=0.5, help="Lagrange Constant to converge to identity matrix", type=float)
        parser.add_argument("--alpha", default=0.1, help="Const. for second loss function", type=float)
        parser.add_argument("--max_epochs", default=200, help="Maximum Number of Epochs", type=int)
        parser.add_argument("--patience", default=2, help="Patience", type=int)
        parser.add_argument("--patience_increase", default=2, help="Patience Increase", type=int)
        parser.add_argument("--improvement_threshold", default=2, help="Improvement Threshold for Patience", type=int)
        parser.add_argument("--save_after", default=0, help="Save after how many Epochs?", type=int)
        parser.add_argument("--epoch_freq", default=1, help="Epoch Frequency", type=int)
        parser.add_argument("--debug", default=False, type=self.str_to_bool, help="Debug Mode")
        parser.add_argument("--retrain", default=False, type=self.str_to_bool, help="Retrain Flag")
        parser.add_argument("--load", default =False, type=self.str_to_bool, help="Load Model to calculate accuracy")
        self.parser=parser

    def str_to_bool(self, text):
        if text.lower() == "true":
            return True
        elif text.lower() == "false":
            return False
        else :
            raise argparse.ArgumentTypeError('Boolean Value Expected')

    def get_parser(self): 
        return self.parser