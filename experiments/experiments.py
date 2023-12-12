import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import os

def log_results(experiment):
    def logged_experiment(path_pattern, classifier, distributions, dataset, njobs, fit_params=None):

        auc_res_path = path_pattern.format('auc')
        acc_res_path = path_pattern.format('accuracy')

        try:
            auc_df = pd.read_csv(auc_res_path, index_col=0)
        except FileNotFoundError:
            auc_df = pd.DataFrame()

        try:
            acc_df = pd.read_csv(acc_res_path, index_col=0)
        except FileNotFoundError:
            acc_df = pd.DataFrame()

        res = experiment(classifier, distributions, dataset, njobs, fit_params=None)

        auc_df[dataset] = {
            classifier.__class__.__name__: f"{res['roc_auc']['mean']:.4f} ± {res['roc_auc']['std']:.4f}"
        }

        acc_df[dataset] = {
            classifier.__class__.__name__: f"{res['accuracy']['mean']:.4f} ± {res['accuracy']['std']:.4f}"
        }

        auc_df.to_csv(auc_res_path)
        acc_df.to_csv(acc_res_path)

        return auc_df, acc_df
    
    return logged_experiment

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

def svm_experiment(X: np.ndarray, y: np.array, classifier, distributions, njobs: int, fit_params = {}) -> (float, float):
    test_fold = [0 if v < len(X) else 1 for v in range(len(X) * 2)]

    search = RandomizedSearchCV(classifier, distributions, cv=PredefinedSplit(
        test_fold=test_fold), refit=False, n_iter=10, scoring=['roc_auc', 'average_precision', 'accuracy'], n_jobs=njobs, error_score='raise', verbose=0)

    res = search.fit(np.concatenate((X, X)), np.concatenate((y, y)), **fit_params)

    cv_results = pd.DataFrame(res.cv_results_)

    return {
        'roc_auc': {
            'mean': np.average(cv_results['mean_test_roc_auc']),
            'std': np.std(cv_results['mean_test_roc_auc'])
        },
        'average_precision': {
            'mean': np.average(cv_results['mean_test_average_precision']),
            'std': np.std(cv_results['mean_test_average_precision'])
        },
        'accuracy': {
            'mean': np.average(cv_results['mean_test_accuracy']),
            'std': np.std(cv_results['mean_test_accuracy'])
        }
    }

@log_results
def compute_baseline(classifier, distributions, dataset, njobs,  fit_params={}):
    dinfo = get_datasets()[dataset]

    X,y = get_dataset_from_path(dinfo)
    res = svm_experiment(X, y, classifier, distributions, njobs)

    return res

@log_results
def compute_federated_experiment(classifier, distributions, dataset, njobs, fit_params=None):
    dinfo = get_datasets()[dataset]

    X,y = get_dataset_from_path(dinfo)

    fit_params = {
        'client_assignment': np.random.choice([0,1], size=len(X)),
        'round_callback': None
    }

    res = svm_experiment(X, y, classifier, distributions, njobs, fit_params)

    return res