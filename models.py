'''
models.py

Model architectures and loading functions are defined here.
'''

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from utils.data import NUM_CLASSES

os.environ['TORCH_HOME'] = 'saved_models'
MODELS_DIR = 'saved_models'


def load_model(model, model_ckpt=None, verbose=True):
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    if model == 'ResNet50':
        module = ResNet50()
    elif model == 'RegularClassifier':
        module = RegularClassifier()
    else:
        raise Exception(f"Unknown model '{model}'. Please refer to models.py for existing modules.")
    if model_ckpt:
        load_checkpoint(module, model_ckpt, verbose)
    return module

def load_checkpoint(model, model_ckpt, verbose=True):
    model_ckpt = os.path.normpath(model_ckpt)
    if os.path.exists(model_ckpt):
        ckpt_path = model_ckpt
    else:
        ckpt_path = os.path.join(MODELS_DIR, model_ckpt)
    if verbose:  print(f"Loading from model checkpoint:  {ckpt_path}")
    model.load_state_dict( torch.load(ckpt_path) )


class ResNet50(nn.Module):

    def __init__(self, freeze=False):
        super(ResNet50, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.backbone.fc = nn.Linear(in_features=2048, out_features=NUM_CLASSES)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        backbone_out = self.backbone(x)
        return self.softmax(backbone_out)


class RegularClassifier(nn.Module):
    def __init__(self):
        super(RegularClassifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 43)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        return self.fc_layers(x)