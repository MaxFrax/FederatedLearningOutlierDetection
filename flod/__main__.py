import os
import logging
from flod.dataset import download_dataset
from flod.features_extraction.load_features import load_features
import json
from sklearn.model_selection import KFold, cross_validate, train_test_split, RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, make_scorer, accuracy_score, classification_report
from flod.classifiers.bsvclassifier import BSVClassifier
import numpy as np
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from joblib import dump
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log", mode='w'),
        logging.StreamHandler()
    ]
)

LOGGER = logging.getLogger(__name__)
CACHE_FOLDER = os.path.abspath('./cache/')
MODELS_FOLDER = os.path.abspath('./cache/models/')

if __name__ == '__main__':
    if not os.path.exists(CACHE_FOLDER):
        LOGGER.info(f'Creating CACHE folder at {CACHE_FOLDER}')
        os.makedirs(CACHE_FOLDER)
    
    dataset_path = download_dataset(CACHE_FOLDER)

    dataset = load_features(CACHE_FOLDER, dataset_path, 100, False, 0.8)

    X = np.array(dataset[['c1', 'c2', 'c3', 'c4']])  # Features
    y = np.array(dataset['is_fall'])  # Labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, stratify=y)

    LOGGER.info(f'Outliers in training {sum(y_train)}/{len(y_train)}')

    pipe = Pipeline([
            ('scaler', MaxAbsScaler()),
            ('reduce_dim', PCA()),
            ('classifier', BSVClassifier())
    ])

    params = {
        'scaler': [StandardScaler(), MaxAbsScaler(), RobustScaler(), QuantileTransformer()],
        'reduce_dim__n_components': range(X.shape[1])[1:],
        'classifier__n_iter': [10],
        'classifier__penalization': np.random.uniform(1, 100, 3),
        'classifier__q': np.random.uniform(0.1, 100, 10)
    }

    scoring = ['precision', 'recall', 'f1']
    cv = StratifiedKFold(n_splits=3, shuffle=True)
    search = GridSearchCV(pipe, params, n_jobs=4, pre_dispatch=4, cv=cv, refit='f1', verbose=5, return_train_score=True, scoring=scoring)

    search.fit(X_train, y_train)

    LOGGER.info(f'Best params: {search.best_params_}')
    LOGGER.info(f'Best score: {search.best_score_}')

    LOGGER.info(f'Score in test data: {search.score(X_test, y_test)}')

    cv_res = pd.DataFrame(search.cv_results_)
    LOGGER.info(cv_res)

    LOGGER.info(classification_report(y_test, search.predict(X_test)))

    if not os.path.exists(MODELS_FOLDER):
        LOGGER.info(f'Creating trained models folder at {MODELS_FOLDER}')
        os.makedirs(MODELS_FOLDER)

    clf = search.best_estimator_['classifier']

    dump_path = os.path.join(MODELS_FOLDER, f'{datetime.now().strftime("%Y%m%d_%H_%M")}.joblib')

    dump(clf, dump_path)