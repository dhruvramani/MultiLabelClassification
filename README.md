# Multi-Label Classification

## Installation 
We recommend the usage of Python 3.5 for the project.
```
pip3 install tensorflow
pip3 install sklearn
pip3 install liac-arff
```
## Running
```
cd model_1_NN/src/
python3 __main__.py
```

## File Structure 
```
.
├── README.md
├── data
│   ├── bookmarks
│   │   ├── bookmarks.arff
│   │   ├── bookmarks.xml
│   │   └── count.txt
│   └── delicious
│       ├── count.txt
│       ├── delicious-test.arff
│       ├── delicious-train.arff
│       ├── delicious.arff
│       └── delicious.xml
└── model_1_NN
    ├── checkpoint
    ├── datasets
    │   └── delicious
    │       └── settings-1
    ├── model_1_NN
    │   └── Default
    ├── results
    ├── resultsmodel_best.ckpt.data-00000-of-00001
    ├── resultsmodel_best.ckpt.index
    ├── resultsmodel_best.ckpt.meta
    ├── src
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── __pycache__
    │   │   ├── config.cpython-35.pyc
    │   │   ├── dataset.cpython-35.pyc
    │   │   ├── eval_performance.cpython-35.pyc
    │   │   ├── network.cpython-35.pyc
    │   │   ├── parser.cpython-35.pyc
    │   │   └── utils.cpython-35.pyc
    │   ├── config.py
    │   ├── dataset.py
    │   ├── eval_performance.py
    │   ├── network.py
    │   ├── parser.py
    │   ├── parser.pyc
    │   ├── run.py
    │   ├── utils.py
    │   └── utils.pyc
    └── stdout
        └── model_1_NN_train.log
```
Results and Logs might vary. 

## Adding Dataset
After adding the dataset, create a file `count.txt` indicating number of features and labels seperated by a new-line.
