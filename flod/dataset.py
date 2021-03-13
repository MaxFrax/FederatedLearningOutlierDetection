import logging
import requests
import os
import zipfile

LOGGER = logging.getLogger(__name__)

DATASET_URL = 'http://sistemic.udea.edu.co/wp-content/uploads/2016/03/SisFall_dataset.zip'
DATASET_FOLDER_NAME = 'SisFall_dataset'
FILENAME = 'SisFall_dataset.zip'

def download_dataset(storage_folder_path: str) -> str:
    file_path = os.path.join(storage_folder_path, FILENAME)
    progress_file_path = os.path.join(storage_folder_path, f"{FILENAME}.inProgress")

    directory_path = os.path.join(storage_folder_path, DATASET_FOLDER_NAME)
    progress_directory_path = os.path.join(storage_folder_path, f"{DATASET_FOLDER_NAME}.inProgress")
    
    if os.path.exists(progress_file_path):
        LOGGER.info(f'Deleting file from previous failed download: f{progress_file_path}')
        os.remove(progress_file_path)

    if not os.path.exists(file_path):
        LOGGER.info(F'Dataset archive not found! Downloading from {DATASET_URL} in {file_path}')
        r = requests.get(DATASET_URL, allow_redirects=True)
        with open(progress_file_path, 'wb') as file:
            file.write(r.content)
        LOGGER.info(f'Download completed! Moving from {progress_file_path} to {file_path}')
        os.rename(progress_file_path, file_path)

    if os.path.exists(progress_directory_path):
        LOGGER.info(f'Found unfinished unzipping! Deleting {progress_directory_path}')
        os.removedirs(progress_directory_path)

    if not os.path.exists(directory_path):
        LOGGER.info(f'Extracting {file_path} in {progress_directory_path}')
        with zipfile.ZipFile(file_path,"r") as zip_ref:
            zip_ref.extractall(progress_directory_path)
        LOGGER.info(f'Unpacking of {directory_path} completed')
        os.rename(progress_directory_path, directory_path)

    return os.path.join(storage_folder_path, DATASET_FOLDER_NAME)
    
