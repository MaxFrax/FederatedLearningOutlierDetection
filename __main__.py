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

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

# create the parser object
parser = argparse.ArgumentParser(
    description='Run the experiments from the paper.')

# add arguments to the parser
parser.add_argument('experiment', type=str, choices=[
                    'baseline_sklearn'], help='The experiment to run.')

parser.add_argument('--njobs', type=int,
                    help='Number of parallel threads to use.', default=-1)

parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING',
                    'ERROR', 'CRITICAL'], help='The logging level.', dest='loglevel', default='INFO')

# parse the arguments
args = parser.parse_args()


def svm_experiment(X: np.ndarray, y: np.array, classifier, distributions, metric: str):
    scorer = make_scorer(roc_auc_score) if metric == 'roc_auc' else make_scorer(
        average_precision_score)

    test_fold = [0 if v < len(X) else 1 for v in range(len(X) * 2)]

    search = RandomizedSearchCV(classifier, distributions, cv=PredefinedSplit(
        test_fold=test_fold), refit=True, n_iter=10, scoring=scorer, n_jobs=args.njobs, error_score='raise')

    res = search.fit(np.concatenate((X, X)), np.concatenate((y, y)))

    cv_results = pd.DataFrame(res.cv_results_)
    cv_results.sort_values('rank_test_score')

    logger.info(
        f"{metric}: {np.average(cv_results['mean_test_score']):.4f} ± {np.std(cv_results['mean_test_score']):.4f}")


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
    logger.info('Running baseline experiment')

    X, y = get_dataset()

    classifier = OneClassSVM(kernel='rbf')
    distributions = dict(nu=uniform(loc=0.2, scale=0.8),
                         gamma=uniform(loc=0, scale=3))

    svm_experiment(X, y, classifier, distributions, 'roc_auc')
    svm_experiment(X, y, classifier, distributions, 'average_precision')


if args.loglevel:
    logger.setLevel(args.loglevel)

if args.experiment == 'baseline_sklearn':
    baseline_sklearn()
