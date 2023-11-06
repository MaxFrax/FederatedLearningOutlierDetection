import argparse
import logging
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, make_scorer, roc_auc_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.svm import OneClassSVM
from scipy.stats import uniform
from sklearn.preprocessing import MinMaxScaler
from flod.classifiers.bsvclassifier import BSVClassifier
from experiments.baseline import svm_experiment

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

# create the parser object
parser = argparse.ArgumentParser(
    description='Run the experiments from the paper.')

# add arguments to the parser
parser.add_argument('experiment', type=str, choices=[
                    'baseline_sklearn', 'baseline_svdd'], help='The experiment to run.')

parser.add_argument('--njobs', type=int,
                    help='Number of parallel threads to use.', default=-1)

parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING',
                    'ERROR', 'CRITICAL'], help='The logging level.', dest='loglevel', default='INFO')

# parse the arguments
args = parser.parse_args()

def get_dataset():
    input_names = []

    for i in range(8):
        input_names.append(f'x{i}')
        input_names.append(f'y{i}')

    data = pd.read_csv(
        './datasets/pen-global-unsupervised-ad.csv', names=input_names+['outlier'])
    data = data.sample(frac=1)

    X = data[input_names]
    y = data['outlier'] == 'o'

    y = [-1 if v else 1 for v in y]

    X = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(X, columns=input_names)

    return X, y


def baseline_sklearn():
    logger.info('Running baseline sklearn experiment')

    X, y = get_dataset()

    classifier = OneClassSVM(kernel='rbf')
    distributions = dict(nu=uniform(loc=0.2, scale=0.8),
                         gamma=uniform(loc=0, scale=3))

    svm_experiment(X, y, classifier, distributions, 'roc_auc', args.njobs)
    svm_experiment(X, y, classifier, distributions, 'average_precision', args.njobs)


def baseline_svdd():
    logger.info('Running baseline svdd experiment')

    X, y = get_dataset()

    classifier = BSVClassifier(normal_class_label=1, outlier_class_label=-1)
    distributions = {'c':uniform(loc=0.2, scale=0.8),'q':uniform(loc=0, scale=3)}

    svm_experiment(X, y, classifier, distributions, 'roc_auc', args.njobs)
    svm_experiment(X, y, classifier, distributions, 'average_precision', args.njobs)


if args.loglevel:
    logger.setLevel(args.loglevel)

if args.experiment == 'baseline_sklearn':
    baseline_sklearn()
elif args.experiment == 'baseline_svdd':
    baseline_svdd()
