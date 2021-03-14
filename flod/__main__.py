import os
import logging
from flod.dataset import download_dataset
from flod.features_extraction.load_features import load_features
import json
from sklearn.model_selection import KFold, cross_validate
from flod.classifiers.bsvclassifier import BSVClassifier
import numpy as np

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

    clf = BSVClassifier(random_state=42)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    
    print(cross_validate(clf, X, y, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1']))