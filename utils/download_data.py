'''
download_data.py

Download the GTSRB dataset. Training images will appear in the data/train directory.
Test images will appear in data/test. (465MBs)

Simply run:  python -m utils.download_data
'''

import requests
import zipfile
import io
import os

TRAIN_DATA_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
TEST_DATA_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
TEST_LABELS_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"
LOCAL_DATA_DIR = "data"

if __name__ == "__main__":
    # create data folder if it doesn't exist
    if not os.path.exists(LOCAL_DATA_DIR):
        os.makedirs(LOCAL_DATA_DIR)
    
    # download the zip files
    print("Downloading data ...")
    train_data_zip = zipfile.ZipFile(io.BytesIO(requests.get(TRAIN_DATA_URL).content))
    test_data_zip = zipfile.ZipFile(io.BytesIO(requests.get(TEST_DATA_URL).content))
    test_labels_zip = zipfile.ZipFile(io.BytesIO(requests.get(TEST_LABELS_URL).content))
    print("Extracting data ...")
    train_data_zip.extractall( os.path.join(LOCAL_DATA_DIR, 'train') )
    test_data_zip.extractall( os.path.join(LOCAL_DATA_DIR, 'test') )
    test_labels_zip.extractall( os.path.join(LOCAL_DATA_DIR, 'test') )
    print("Done")

