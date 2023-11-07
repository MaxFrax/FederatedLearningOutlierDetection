from ast import List
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import logging
import sys
import numpy as np
import pandas as pd
import os
from sklearn.metrics import average_precision_score, make_scorer, roc_auc_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.svm import OneClassSVM
from scipy.stats import uniform
from sklearn.preprocessing import MinMaxScaler
from flod.classifiers.bsvclassifier import BSVClassifier
from experiments.baseline import svm_experiment
import glob
from tqdm import tqdm

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

def get_datasets():
    return [{ 
        'name': os.path.splitext(os.path.basename(dataset))[0].replace('-unsupervised-ad',''), 
        'path': dataset
    } for dataset in glob.glob('./datasets/*.csv')]


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
    logger.info('Running baseline sklearn experiment')

    classifier = OneClassSVM(kernel='rbf')
    distributions = dict(nu=uniform(loc=0.2, scale=0.8),
                         gamma=uniform(loc=0, scale=3))

    auc_table = {}
    avg_table = {}

    for dinfo in tqdm(get_datasets()):

        tqdm.write(f'Running experiment for {dinfo["name"]}')
        X,y = get_dataset_from_path(dinfo['path'])
        auc_res = svm_experiment(X, y, classifier, distributions, 'roc_auc', args.njobs)
        auc_table[dinfo['name']] = {
            classifier.__class__.__name__: f'{auc_res[0]:.4f} ± {auc_res[1]:.4f}'
        }

        avg_res = svm_experiment(X, y, classifier, distributions, 'average_precision', args.njobs)
        avg_table[dinfo['name']] = {
            classifier.__class__.__name__: f'{avg_res[0]:.4f} ± {avg_res[1]:.4f}'
        }

        auc_df = pd.DataFrame(auc_table)
        avg_df = pd.DataFrame(avg_table)

        auc_df.to_csv('sklearn_auc.csv')
        avg_df.to_csv('sklearn_avg.csv')

    auc_df = pd.DataFrame(auc_table)
    avg_df = pd.DataFrame(avg_table)
    print('AUC')
    print(auc_df)
    print('Average Precision')
    print(avg_df)


def baseline_svdd():
    logger.info('Running baseline svdd experiment')

    classifier = BSVClassifier(normal_class_label=1, outlier_class_label=-1)
    distributions = {'c':uniform(loc=0.2, scale=0.8),'q':uniform(loc=0, scale=3)}

    auc_table = {}
    avg_table = {}

    for dinfo in tqdm(get_datasets()):

        tqdm.write(f'Running experiment for {dinfo["name"]}')
        X,y = get_dataset_from_path(dinfo['path'])
        auc_res = svm_experiment(X, y, classifier, distributions, 'roc_auc', args.njobs)
        auc_table[dinfo['name']] = {
            classifier.__class__.__name__: f'{auc_res[0]:.4f} ± {auc_res[1]:.4f}'
        }

        avg_res = svm_experiment(X, y, classifier, distributions, 'average_precision', args.njobs)
        avg_table[dinfo['name']] = {
            classifier.__class__.__name__: f'{avg_res[0]:.4f} ± {avg_res[1]:.4f}'
        }

        auc_df = pd.DataFrame(auc_table)
        avg_df = pd.DataFrame(avg_table)

        auc_df.to_csv('svdd_auc.csv')
        avg_df.to_csv('svdd.csv')

    auc_df = pd.DataFrame(auc_table)
    avg_df = pd.DataFrame(avg_table)
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
