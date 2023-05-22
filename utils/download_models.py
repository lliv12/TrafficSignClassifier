'''
download_models.py

Download pretrained model for GTSRB classification. Model size is 526 MB,
gets ~87% test set accuracy.

Simply run:  python -m utils.download_models
'''

from huggingface_hub import hf_hub_url
import os
import requests
from models import MODELS_DIR

REPO_ID = "lliv12/regular_classifier_best"
FILENAME = "regular_classifier_best.pt"

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    print("Downloading model ...")
    url = hf_hub_url(repo_id=REPO_ID, filename=FILENAME)
    response = requests.get(url)
    with open(os.path.join(MODELS_DIR, FILENAME), "wb") as f:
        f.write(response.content)
    print("Done")