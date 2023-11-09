import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

def svm_experiment(X: np.ndarray, y: np.array, classifier, distributions, njobs: int) -> (float, float):
    test_fold = [0 if v < len(X) else 1 for v in range(len(X) * 2)]

    search = RandomizedSearchCV(classifier, distributions, cv=PredefinedSplit(
        test_fold=test_fold), refit=False, n_iter=10, scoring=['roc_auc', 'average_precision'], n_jobs=njobs, error_score='raise', verbose=0)

    res = search.fit(np.concatenate((X, X)), np.concatenate((y, y)))

    cv_results = pd.DataFrame(res.cv_results_)

    return {
        'roc_auc': {
            'mean': np.average(cv_results['mean_test_roc_auc']),
            'std': np.std(cv_results['mean_test_roc_auc'])
        },
        'average_precision': {
            'mean': np.average(cv_results['mean_test_average_precision']),
            'std': np.std(cv_results['mean_test_average_precision'])
        }
    }

def compute_baseline(auc_res_path, avg_res_path, classifier, distributions, dataset, njobs):
    try:
        auc_df = pd.read_csv(auc_res_path, index_col=0)
    except FileNotFoundError:
        auc_df = pd.DataFrame()

    try:
        avg_df = pd.read_csv(avg_res_path, index_col=0)
    except FileNotFoundError:
        avg_df = pd.DataFrame()

    dinfo = get_datasets()[dataset]

    X,y = get_dataset_from_path(dinfo)
    res = svm_experiment(X, y, classifier, distributions, njobs)

    auc_df[dataset] = {
        classifier.__class__.__name__: f"{res['roc_auc']['mean']:.4f} ± {res['roc_auc']['std']:.4f}"
    }
    avg_df[dataset] = {
        classifier.__class__.__name__: f"{res['average_precision']['mean']:.4f} ± {res['average_precision']['std']:.4f}"
    }

    auc_df.to_csv(auc_res_path)
    avg_df.to_csv(avg_res_path)

    return auc_df, avg_df

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
    for dataset in glob.glob('./datasets/*.csv'):
        datasets[os.path.splitext(os.path.basename(dataset))[0].replace('-unsupervised-ad','')] = dataset

    return datasets