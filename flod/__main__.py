import os
import logging
from flod.dataset import download_dataset

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
    
    download_dataset(CACHE_FOLDER)