import glob
import os
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import logging

import os

LOGGER = logging.getLogger(__name__)

def get_dataset_from_path(path):

    data = pd.read_csv(path, header=0)
    data = data.sample(frac=1)

    data.columns = [f'x{i}' for i in range(len(data.columns) - 1)] + ['outlier']

    X = data[data.columns[:-1]]
    y = data['outlier'] == 'o'

    y = [-1 if v else 1 for v in y]

    X = MinMaxScaler().fit_transform(X)

    return X, y


def get_datasets():
    datasets = {}
    path = ''
    if os.path.exists('./datasets'):
        path = './datasets'
    elif os.path.exists('../datasets'):
        path = '../datasets'

    for dataset in glob.glob(os.path.join(path, '*.csv')):
        datasets[os.path.splitext(os.path.basename(dataset))[0].replace('-unsupervised-ad','')] = dataset

    return datasets

def unsupervised_experiment(X: np.ndarray, y: np.array, classifier, distributions, njobs: int, fit_params = {}) -> Tuple[float, float]:
    accuracies = []
    aucs = []

    # TODO parallelizzare
    for _ in range(10):
        # Draw from distributions
        if distributions.get('C'):
            drawn_samples = distributions['C'].rvs()
            classifier.C = drawn_samples
        else:
            drawn_samples = distributions['c'].rvs()
            classifier.c = drawn_samples

        # Fit the model over the whole dataset
        classifier.fit(X, y, **fit_params)

        # Compute the metrics over the whole dataset ROC AUC and Accuracy
        y_pred = classifier.predict(X)
        aucs.append(roc_auc_score(y, classifier.score_samples(X)))
        accuracies.append(accuracy_score(y, y_pred))

    return {
        'roc_auc': {
            'mean': np.average(aucs),
            'std': np.std(aucs)
        },
        'accuracy': {
            'mean': np.average(accuracies),
            'std': np.std(accuracies)
        }
    }

def nested_crossval_experiment(X: np.ndarray, y: np.array, classifier, distributions, njobs: int, fit_params = {}) -> (float, float):
    # Declare the inner and outer cross-validation strategies
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=941703)
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=941703)

    # Inner cross-validation for parameter search
    model = RandomizedSearchCV(classifier, distributions, cv=inner_cv, n_iter=10, scoring=['roc_auc', 'accuracy'], n_jobs=njobs, error_score='raise', verbose=0, refit='accuracy')

    # Outer cross-validation to compute the testing score
    test_score = cross_validate(model, X, y, cv=outer_cv, n_jobs=njobs, scoring=['roc_auc', 'accuracy'], fit_params=fit_params, verbose=0)

    return {
        'roc_auc': {
            'mean': np.average(test_score['test_roc_auc']),
            'std': np.std(test_score['test_roc_auc'])
        },
        'accuracy': {
            'mean': np.average(test_score['test_accuracy']),
            'std': np.std(test_score['test_accuracy'])
        }
    }

def compute_baseline(classifier, distributions, dataset, njobs, iid, eval_technique, fit_params={}):
    dinfo = get_datasets()[dataset]

    X,y = get_dataset_from_path(dinfo)
    if eval_technique == 'unsupervised':
        res = unsupervised_experiment(X, y, classifier, distributions, njobs)
    else:
        res = nested_crossval_experiment(X, y, classifier, distributions, njobs)

    return res

def compute_federated_experiment(classifier, distributions, dataset, njobs, iid, eval_technique, fit_params=None):
    np.random.seed(941703)

    dinfo = get_datasets()[dataset]
    clients = classifier.total_clients

    X,y = get_dataset_from_path(dinfo)

    if iid == 'iid':
        try:
            assignment = np.load(f'iid_assignment_{dataset}_{clients}.npy', allow_pickle=True)
        except FileNotFoundError:
            LOGGER.warning(f'Could not find iid assignment file for {clients} clients. Generating a new one.')
            assignment = np.random.choice(list(range(clients)), size=len(X))
            assignment.dump(f'iid_assignment_{dataset}_{clients}.npy')
    else:
        try:
            assignment = np.load(f'non_iid_assignment_{dataset}_{clients}.npy', allow_pickle=True)
        except FileNotFoundError:
            LOGGER.warning(f'Could not find non iid assignment file for {clients} clients. Generating a new one.')
            assignment = KMeans(n_clusters=clients).fit_predict(X)
            assignment.dump(f'non_iid_assignment_{dataset}_{clients}.npy')

    fit_params = {
        'client_assignment': assignment,
        'round_callback': None
    }

    if eval_technique == 'unsupervised':
        res = unsupervised_experiment(X, y, classifier, distributions, njobs, fit_params)
    else:
        res = nested_crossval_experiment(X, y, classifier, distributions, njobs, fit_params)

    return res