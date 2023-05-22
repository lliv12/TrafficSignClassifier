'''
test.py

Provides validate() function for evaluating a model checkpoint on a given
dataset. Run this script to evaluate a model checkpoint on the test set.
Use this schema:

python test.py <model> --model_ckpt --batch_size

  + model:      name of the model architecture (module class from models.py)
  --model_ckpt: name of the model checkpoint / absolute path to the checkpoint
  --batch_size: what batch size to use for test set evaluation
'''

import argparse
import torch
from models import load_model
from utils.data import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def validate(model, data_loader, loss_func=None, verbose=False):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(data_loader, desc='Validating', unit='batch')
    for batch in progress_bar:
        x, y = batch
        pred = model(x)
        if loss_func:
            val_loss += loss_func(pred, y).item()
        _, pred_idx = torch.max(pred.data, 1)
        _, y_idx = torch.max(y, 1)
        total += y.size(0)
        correct += (pred_idx == y_idx).sum().item()
        progress_bar.set_description(f'Validating ({correct}/{total} correct)')
    if loss_func:  val_loss /= len(data_loader)
    val_acc = correct / total

    if verbose:
        print(f"accuracy:  {np.around(val_acc, 4)}")
        if val_acc:  print(f"val loss:  {np.around(val_loss, 4)}")

    return val_loss, val_acc if (loss_func) else val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='the model to perform validation on')
    parser.add_argument('--model_ckpt', help='the model checkpoint')
    parser.add_argument('--batch_size', default=16, help='batch size to use')
    args = parser.parse_args()

    model = load_model(args.model, args.model_ckpt)
    data_loader = DataLoader(load_dataset('test'), batch_size=args.batch_size)
    validate(model, data_loader, verbose=True)