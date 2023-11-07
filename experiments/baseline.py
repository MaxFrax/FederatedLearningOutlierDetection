import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

def svm_experiment(X: np.ndarray, y: np.array, classifier, distributions, metric: str, njobs: int) -> (float, float):
    test_fold = [0 if v < len(X) else 1 for v in range(len(X) * 2)]

    search = RandomizedSearchCV(classifier, distributions, cv=PredefinedSplit(
        test_fold=test_fold), refit=True, n_iter=10, scoring=metric, n_jobs=njobs, error_score='raise', verbose=0)

    res = search.fit(np.concatenate((X, X)), np.concatenate((y, y)))

    cv_results = pd.DataFrame(res.cv_results_)
    cv_results.sort_values('rank_test_score')

    return np.average(cv_results['mean_test_score']), np.std(cv_results['mean_test_score'])