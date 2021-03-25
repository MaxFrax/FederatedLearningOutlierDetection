import os
import logging
from flod.dataset import download_dataset
from flod.features_extraction.load_features import load_features
import json
from sklearn.model_selection import KFold, cross_validate, train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
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

    clf = BSVClassifier(c=0.03142918568673425, penalization=230, q=51)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=100, test_size=200, stratify=y, shuffle=True)

    LOGGER.info(f'Training size {len(X_train)} and test size {len(X_test)}. Outliers in training {sum(y_train)}')

    clf.fit(X_train, y_train)

    LOGGER.info(f'Sum betas {sum(clf.betas_)} Count neg {len([x for x in clf.betas_ if x < 0])}')

    y_pred = clf.predict(X_test)

    LOGGER.info(f'Predicted. Outliers in test sample are {sum(y_test)}')

    print(confusion_matrix(y_test, y_pred))

    print(f'Precision {precision_score(y_test, y_pred)}')
    print(f'Recall {recall_score(y_test, y_pred)}')
    print(f'F1 {f1_score(y_test, y_pred)}')


    parameters = {
        'c': sp_randFloat(0, 1),
        'q': sp_randInt(0, 200),
        'penalization' : sp_randInt(0, 400),
        'n_iter': [10]
    }

    #Come viene calcolato lo score?
    model = BSVClassifier()
    randm_src = RandomizedSearchCV(estimator=model, param_distributions = parameters,
                               cv = 2, n_iter = 5, n_jobs=-1)
    randm_src.fit(X_train, y_train)

    print(" Results from Random Search " )
    print("\n The best estimator across ALL searched params:\n", randm_src.best_estimator_)
    print("\n The best score across ALL searched params:\n", randm_src.best_score_)
    print("\n The best parameters across ALL searched params:\n", randm_src.best_params_)