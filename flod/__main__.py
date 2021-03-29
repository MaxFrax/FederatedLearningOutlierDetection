import os
import logging
from flod.dataset import download_dataset
from flod.features_extraction.load_features import load_features
import json
from sklearn.model_selection import KFold, cross_validate, train_test_split, RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, make_scorer, accuracy_score
from flod.classifiers.bsvclassifier import BSVClassifier
import numpy as np
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

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

if __name__ == '__main__':
    if not os.path.exists(CACHE_FOLDER):
        LOGGER.info(f'Creating CACHE folder at {CACHE_FOLDER}')
        os.makedirs(CACHE_FOLDER)
    
    dataset_path = download_dataset(CACHE_FOLDER)

    dataset = load_features(CACHE_FOLDER, dataset_path)

    X = np.array(dataset[['c1', 'c2', 'c3', 'c4']])  # Features
    y = np.array(dataset['is_fall'])  # Labels

    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=900, test_size=100, stratify=y, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=4500, test_size=500, stratify=y, shuffle=True)

    LOGGER.info(f'Outliers in training {sum(y_train)}/{len(y_train)}')

    max_distance = 0
    for i in X_train:
        for j in X_train:
            dist = np.linalg.norm(i-j, ord=2)
            if dist > max_distance:
                max_distance = dist

    LOGGER.info(f'Max distance among training set points {max_distance}')

    zetas = [z for z in range(len(X_train))]

    parameters = {
            #'c': [1 / (sum(y_train) * z) for z in random.sample(zetas, 20)],
            'c': [1 / (sum(y_train) * len(y_train))],
            'q': [(k+1) / (max_distance**2) for k in range(10)],
            'penalization' : [100],
            'n_iter': [10]
        }

    cv = StratifiedKFold(n_splits=2, shuffle=True)
    model = BSVClassifier()

    randm_src = GridSearchCV(estimator=model, param_grid = parameters,
                            cv = cv, n_jobs=-1, scoring=make_scorer(f1_score), verbose=10)
    randm_src.fit(X_train, y_train)

    print(" Results from Random Search " )
    print("\n The best estimator across ALL searched params:\n", randm_src.best_estimator_)
    print("\n The best score across ALL searched params:\n", randm_src.best_score_)
    print("\n The best parameters across ALL searched params:\n", randm_src.best_params_)

    clf = BSVClassifier(**randm_src.best_params_)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f'Precision {precision_score(y_test, y_pred)}')
    print(f'Recall {recall_score(y_test, y_pred)}')
    print(f'F1 {f1_score(y_test, y_pred)}')
    print(f'Accuracy {accuracy_score(y_test, y_pred)}')