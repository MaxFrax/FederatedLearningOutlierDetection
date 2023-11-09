import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import logging
import sys
import numpy as np
import pandas as pd
import os
from sklearn.svm import OneClassSVM
from scipy.stats import uniform
from sklearn.preprocessing import MinMaxScaler
from flod.classifiers.bsvclassifier import BSVClassifier
from experiments.baseline import svm_experiment
import glob

def get_datasets():
    datasets = {}
    for dataset in glob.glob('./datasets/*.csv'):
        datasets[os.path.splitext(os.path.basename(dataset))[0].replace('-unsupervised-ad','')] = dataset

    return datasets

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

parser.add_argument('dataset', type=str, choices=get_datasets().keys(), help='The dataset to run the experiment on.')

# parse the arguments
args = parser.parse_args()


def get_dataset_from_path(path):

    data = pd.read_csv(path, header=0)
    data = data.sample(frac=1)

    data.columns = [f'x{i}' for i in range(len(data.columns) - 1)] + ['outlier']

    X = data[data.columns[:-1]]
    y = data['outlier'] == 'o'

    y = [-1 if v else 1 for v in y]

    X = MinMaxScaler().fit_transform(X)

    return X, y

def baseline_sklearn():
    auc_res_path = 'sklearn_auc.csv'
    avg_res_path = 'sklearn_avg.csv'

    logger.info('Running baseline sklearn experiment on %s', args.dataset)

    classifier = OneClassSVM(kernel='rbf')
    distributions = dict(nu=uniform(loc=0.2, scale=0.8),
                         gamma=uniform(loc=0, scale=3))
    
    try:
        auc_df = pd.read_csv(auc_res_path, index_col=0)
    except FileNotFoundError:
        auc_df = pd.DataFrame()

    try:
        avg_df = pd.read_csv(avg_res_path, index_col=0)
    except FileNotFoundError:
        avg_df = pd.DataFrame()

    dinfo = get_datasets()[args.dataset]

    X,y = get_dataset_from_path(dinfo)
    res = svm_experiment(X, y, classifier, distributions, args.njobs)

    auc_df[args.dataset] = {
        classifier.__class__.__name__: f"{res['roc_auc']['mean']:.4f} ± {res['roc_auc']['std']:.4f}"
    }
    avg_df[args.dataset] = {
        classifier.__class__.__name__: f"{res['average_precision']['mean']:.4f} ± {res['average_precision']['std']:.4f}"
    }

    auc_df.to_csv('sklearn_auc.csv')
    avg_df.to_csv('sklearn_avg.csv')

    print('AUC')
    print(auc_df)
    print('Average Precision')
    print(avg_df)

def baseline_svdd():
    auc_res_path = 'svdd_auc.csv'
    avg_res_path = 'svdd_avg.csv'

    logger.info('Running baseline sklearn experiment on %s', args.dataset)

    classifier = BSVClassifier(normal_class_label=1, outlier_class_label=-1)
    distributions = {'c':uniform(loc=0.2, scale=0.8),'q':uniform(loc=0, scale=3)}
    
    try:
        auc_df = pd.read_csv(auc_res_path, index_col=0)
    except FileNotFoundError:
        auc_df = pd.DataFrame()

    try:
        avg_df = pd.read_csv(avg_res_path, index_col=0)
    except FileNotFoundError:
        avg_df = pd.DataFrame()

    dinfo = get_datasets()[args.dataset]

    X,y = get_dataset_from_path(dinfo)
    res = svm_experiment(X, y, classifier, distributions, args.njobs)

    auc_df[args.dataset] = {
        classifier.__class__.__name__: f"{res['roc_auc']['mean']:.4f} ± {res['roc_auc']['std']:.4f}"
    }
    avg_df[args.dataset] = {
        classifier.__class__.__name__: f"{res['average_precision']['mean']:.4f} ± {res['average_precision']['std']:.4f}"
    }

    auc_df.to_csv('sklearn_auc.csv')
    avg_df.to_csv('sklearn_avg.csv')

    print('AUC')
    print(auc_df)
    print('Average Precision')
    print(avg_df)


if args.loglevel:
    logger.setLevel(args.loglevel)

if args.experiment == 'baseline_sklearn':
    baseline_sklearn()
elif args.experiment == 'baseline_svdd':
    baseline_svdd()
