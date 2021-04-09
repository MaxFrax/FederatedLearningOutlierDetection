import pandas as pd
import os
import logging
import json
from tqdm import tqdm
from flod.features_extraction.sisfallfeaturesextraction import SisFallFeaturesExtraction
import pickle

LOGGER = logging.getLogger(__name__)

LABELS_PATH = 'labels.json'
FEATURES_FOLDER = 'features'

def load_features(cache_folder: str, dataset_path: str, window_size: int, filtered: bool, overlap: float) -> pd.DataFrame:
    cache_features_path = os.path.join(cache_folder, FEATURES_FOLDER)
    labels = {}

    if not os.path.exists(cache_features_path):
        LOGGER.info(f'Creating features cache folder at {cache_features_path}')
        os.makedirs(cache_features_path )

    LOGGER.info(f'Loading labels from {LABELS_PATH}')
    with open(LABELS_PATH) as l_file:
        labels = json.load(l_file)

    dataset = None
    
    path_to_walk = os.path.join(dataset_path, 'SisFall_dataset')
    LOGGER.info(f'Looking for data in subfolders of {path_to_walk}')
    for path, folder_list, file_list in tqdm(os.walk(path_to_walk)):
            for file in file_list:

                if file in labels:
                    
                    # If we don't have the labels for this samples, let's skip them
                    if labels[file] is None:
                        continue

                    base_name = f'{file}_{str(window_size)}_{"f" if filtered else ""}_{str(overlap).replace(".","_")}'
                    progress_filename = f'{base_name}.inProgress'
                    progress_feature_path = os.path.join(cache_features_path, progress_filename)
                    feature_path = os.path.join(cache_features_path, base_name)
                    
                    if os.path.exists(progress_feature_path):
                        LOGGER.info(f'Deleting file from previous failed caching: f{progress_feature_path}')
                        os.remove(progress_file_path)

                    begin, end = labels[file]['begin'], labels[file]['end']

                    if os.path.exists(feature_path):
                        LOGGER.info(f'Loading cached feature: f{feature_path}')
                        with open(feature_path, 'rb') as f:
                            extraction = pickle.load(f)
                    else:
                        file_path = os.path.join(path, file)
                        
                        extraction = SisFallFeaturesExtraction(path=file_path, fall_begin_sample=begin, fall_end_sample=end)
                        extraction.compute_features(window_size, filtered, overlap)

                        LOGGER.info(f'Caching features {progress_feature_path}')
                        with open(progress_feature_path, 'wb') as f:
                            pickle.dump(extraction, f)
                        
                        os.rename(progress_feature_path, feature_path)
                    
                    if dataset is None:
                        dataset = extraction.features
                    else:
                        dataset = pd.concat([dataset, extraction.features])

    return dataset