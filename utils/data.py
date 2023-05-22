'''
data.py

Modules for the GTSRB dataset, and some functions for loading image data. Use
this schema to preview image data:

python utils/data.py --filepath --im_class --instance --resolution
  --filepath:   filepath to the image
  --im_class:   which image class you want to visualualize (Ex: 'stop_sign')
  --instance:   which image instance you want to visualize
  --resolution: which instance resolution you want to visualualize (0-29)
'''

import argparse
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import random
import csv
import matplotlib.pyplot as plt

TRAIN_PATH = os.path.normpath("data/train/GTSRB/Final_Training/Images")
TEST_PATH = os.path.normpath("data/test/GTSRB/Final_Test/Images")
TEST_LABELS_PATH = os.path.normpath("data/test/GT-final_test.csv")
DATASET_DICT = 'dataset_dict.txt'
NUM_CLASSES = 43


class GTSRBDataset(Dataset):
    '''
    base_dir:  the base directory where images are located
    soft_load:  whether or not to load the dataset softly (without pre-loading tensored data)
    resolution:  the resolution of images to load
    '''
    def __init__(self, base_dir, soft_load=True, resolution=256, limit=None):
        super(GTSRBDataset, self).__init__()
        self.soft_load = soft_load
        self.resolution = (resolution, resolution)
        self.file_paths = self._get_file_paths(base_dir)
        if not limit:
            limit = len(self.file_paths)
        else:
            self.file_paths = self.file_paths[:limit]

        if not soft_load:
            self.x = torch.stack([load_ppm_image(d, self.resolution) for d in self.file_paths])
            self.y = torch.stack([self._load_img_label_onehot(path) for path in self.file_paths]).squeeze()

    def __getitem__(self, idx):
        if self.soft_load:
            if type(idx) == int:
                return (self._load_img(idx), self._load_img_label_onehot(self.file_paths[idx]).squeeze())
            elif type(idx) in [list, range]:
                x = torch.stack([self._load_img(i) for i in idx])
                y = torch.stack([self._load_img_label_onehot(self.file_paths[i]) for i in idx]).squeeze()
                return (x, y)
        else:
            return (self.x[idx], self.y[idx])

    def __len__(self):
        if self.soft_load:
            return len(self.file_paths)
        else:
            return self.x.shape[0]

    def _get_file_paths(self, parent_dir):
        directories = []
        for root, _, files in os.walk(parent_dir):
            if files:
                directories.extend([os.path.join(root, f) for f in files if f.split('.')[-1] == 'ppm'])
        return directories

    def _load_img(self, idx):
        return load_ppm_image(self.file_paths[idx], self.resolution)

    def _load_img_label_onehot(self, dir):
        img_class = int(dir.split(os.sep)[-2])
        return nn.functional.one_hot(torch.tensor([img_class]), num_classes=NUM_CLASSES).float()

    def display_image(self, idx):
        image = self._load_img(idx)
        plt.imshow(image.permute(1, 2, 0))
        plt.axis('off')
        plt.show()

class GTSRBTestDataset(GTSRBDataset):
    def __init__(self, **kwargs):
        super(GTSRBTestDataset, self).__init__(**kwargs)
        self.labels_dict = self._load_labels_dict()

    def _load_img_label_onehot(self, dir):
        img_class = self.labels_dict[os.path.basename(dir)]
        return nn.functional.one_hot(torch.tensor([img_class]), num_classes=NUM_CLASSES).float()

    def _get_file_paths(self, parent_dir):
        return [os.path.join(parent_dir, file_name) for file_name in os.listdir(parent_dir) if file_name.split('.')[-1] == 'ppm']

    def _load_labels_dict(self):
        annotations = {}
        with open(TEST_LABELS_PATH, 'r') as file:
            reader = csv.DictReader(file, delimiter=';')
            for row in reader:
                annotations[row['Filename']] = int(row['ClassId'])
        return annotations


def load_ppm_image(dir, resolution=(256, 256), from_pil=False, normalize=True):
    image = Image.open(dir)
    image = image.resize(resolution)
    if not from_pil:
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        if normalize:  image_tensor /= 255.0
        return image_tensor
    return image

def load_dataset(dataset, soft_load=True, resolution=256, limit=None):
    if dataset == 'train':
        return GTSRBDataset(TRAIN_PATH, soft_load, resolution, limit)
    elif dataset == 'test':
        return GTSRBTestDataset(base_dir=TEST_PATH, soft_load=soft_load, resolution=resolution, limit=limit)

def load_dataset_dict():
    dataset_dict = dict()
    file = open(DATASET_DICT, 'r')
    for line in file.readlines():
        line_split = line.split(': ')
        dataset_dict[line_split[0]] = line_split[1].replace('\n', '')
    file.close()
    return dataset_dict

def img_tensor_to_pil(image):
    image_np = image.permute(1, 2, 0).numpy()
    scaled_image = (image_np / np.max(image_np)) * 255
    return Image.fromarray(np.uint8(scaled_image))

def show_image(image):
    image.show()


'''
filepath:    whether or not to return the file path to this image instead of the image itself (bool)
from_pil:    whether or not to load a PIL image instead of a tensor.
dataset:     which dataset to pull the image from  ('Train' or 'Test'). If 'Test', then only the instance parameter is looked at.
im_class:    the class of the image (str)
instance:    which image instance to load (int)
resolution:  which resolution of this image instance to load (int; generally between [0, 29])

NOTE:  'None' means it samples a random value by default
'''
def get_image(filepath=False, from_pil=True, dataset=None, im_class=None, instance=None, resolution=None):
    assert type(filepath) == bool
    assert dataset in [None, 'Train', 'Test']
    if dataset == None:  dataset = 'Train'

    if dataset == 'Train':
        file_path = TRAIN_PATH

        dataset_dict_reverse = {v: k for k, v in load_dataset_dict().items()}
        classes = list(dataset_dict_reverse.keys())
        if im_class == None:  im_class = random.choice(classes)
        assert im_class in classes
        file_path = os.path.join(file_path, dataset_dict_reverse[im_class])

        num_instances = int(os.listdir(file_path)[-2].split('_')[0])
        if instance == None:  instance = random.randint(0, num_instances)
        assert instance >= 0  and  instance <= num_instances
        file_instance = str(instance).zfill(5)

        highest_res = [f for f in os.listdir(file_path) if f.split('_')[0] == file_instance]
        highest_res = int(highest_res[-1].split('_')[1].split('.')[0])
        if resolution == None:  resolution = random.randint(0, highest_res)
        assert resolution >= 0  and  resolution <= highest_res
        file_instance = file_instance + '_' + str(highest_res).zfill(5) + '.ppm'

        file_path = os.path.join(file_path, file_instance)
    elif dataset == 'Test':
        file_instances = os.listdir(TEST_PATH)
        if instance == None:
            file_path = os.path.join(TEST_PATH, random.choice(file_instances))
        else:
            assert instance >= 0  and  instance <= len(file_instances)
            file_path = os.path.join(TEST_PATH, str(instance).zfill(5) + '.ppm')

    if filepath:
        return file_path
    else:
        return load_ppm_image(file_path, from_pil=from_pil)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', help="filepath to the image")
    parser.add_argument('--im_class', help="which image class you want to visualualize (Ex: 'stop_sign')")
    parser.add_argument('--instance', type=int, help="which image instance you want to visualize")
    parser.add_argument('--resolution', type=int, help="which instance resolution you want to visualualize (0-29)")
    args = parser.parse_args()

    if args.filepath:
        if (args.im_class or args.instance or args.resolution):
            raise Exception("Either 'filepath' or image attributes may be specified. Not both.")
        else:
            image = get_image(filepath=args.filepath, from_pil=True)
    else:
        image = get_image(filepath=False, from_pil=True, im_class=args.im_class, instance=args.instance, resolution=args.resolution)

    show_image(image)
