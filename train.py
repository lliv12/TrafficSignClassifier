'''
train.py

Code for training and logging the model. Use this schema to run training:

python train.py <model_type> --model_ckpt --model_name --pipeline --e --b --val_frac --save_best --clear_tb_logs --verbose
  + model_type:    the type of model to train (Ex: 'ResNet50')
  --model_ckpt:    (optional) start from a pretrained model checkpoint
  --model_name:    name of the model. Will default to <model_type>
  --pipeline:      (optional) which data augmentation pipeline to use (refer to pipelines.py)
  --e:             how many epochs to run training for
  --b:             the batch size to use for training
  --val_frac:      the fraction of the training data to use for validation (Ex: 0.2)
  --save_best:     (default: True) save the best model checkpoint rather than the last model
  --clear_tb_logs: (default: True) clear tensorboard logs for this model before starting the run
  --verbose:       (default: True) log model performance to the console

Steps for running Tensorboard:
    (With Tensorboard extension in VS Code):
    1)  Ctrl + Shift + P  to view commands
    2)  Select "Launch Tensorboard" >> "Use current working directory" (or select folder where logs is contained)
    NOTE:  Refresh a couple of times until the graphs show up

    (Without Tensorboard extension):
    1)  Open a separate terminal and run "tensorboard --logdir logs"
    2)  Go to the link provided in your web browser (Ex:  "http://localhost:6006/")
'''

import os
import argparse
from models import MODELS_DIR, load_model
from pipelines import load_pipeline
from test import validate
from utils.data import load_dataset
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
import numpy as np

LOGS_DIR = 'logs'


def train(model_type, model_name, pipeline=None, model_ckpt=None, num_epochs=10, batch_size=16, lr=1e-3, val_frac=0.2, save_best=True, clear_tb_logs=True, verbose=True):
    model = load_model(model_type, model_ckpt, verbose=verbose)
    save_dir = os.path.join(MODELS_DIR, model_name + '.pt')
    logs_dir = os.path.join(LOGS_DIR, model_name)

    if clear_tb_logs and os.path.exists(logs_dir):
        for file_name in os.listdir(logs_dir):
            if file_name.startswith("events.out.tfevents."):
                if verbose:  print("removing previous tensorboard logs ...")
                file_path = os.path.join(logs_dir, file_name)
                os.unlink(file_path)

    pipeline = load_pipeline(pipeline) if pipeline else None

    print("\nLoading dataset ...")
    train_dataset, val_dataset = random_split(load_dataset('train'), [1-val_frac, val_frac])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(load_dataset('test'), batch_size=batch_size)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    writer = SummaryWriter(logs_dir)

    best_val_acc = 0.0

    global_step = 0
    for e in range(num_epochs):
        if verbose:  print("\n" + 20*"-" + f" EPOCH {e} " + 20*"-")
        model.train()
        with tqdm(total=len(train_loader), position=0, leave=True) as pbar:
            for batch in train_loader:
                x, y = batch
                if pipeline:
                    x = pipeline(x)
                pred = model(x)

                loss = loss_func(pred, y)
                writer.add_scalar('loss/train', loss.item(), global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                pbar.update(1)
        model.eval()
        val_loss, val_acc = validate(model, val_loader, loss_func)
        writer.add_scalar('loss/val', val_loss, e)
        writer.add_scalar('acc/val', val_acc, e)

        if not save_best:
            torch.save(model.state_dict(), save_dir)
        elif val_acc > best_val_acc:
            print(f"new best model found; previous best:  {best_val_acc}")
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir)
        if verbose:
            print(f"val accuracy:  {np.around(val_acc, 4)}   val loss:  {np.around(val_loss, 4)}")

    # evaluate the best model on the test set
    validate(load_model(model_type, save_dir), test_loader, verbose=True)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', help="the type of model to train (Ex: 'ResNet50')")
    parser.add_argument('--model_ckpt', help="start training from this model checkpoint")
    parser.add_argument('--model_name', help="an alternative name for the model other than <model_type>")
    parser.add_argument('--pipeline', help="specify a data augmentation pipeline to use (refer to pipelines.py)")
    parser.add_argument('--e', type=int, default=10, help='how many epochs to run training for')
    parser.add_argument('--b', type=int, default=16, help='the batch size to use for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--val_frac', type=float, default=0.2, help='the portion of the training data to use for validation')
    parser.add_argument('--save_best', default=True, help='save the best model checkpoint rather than the last model')
    parser.add_argument('--clear_tb_logs', default=True, help='clear tensorboard logs for this model before starting the run')
    parser.add_argument('--verbose', default=True, help='log model performance to the console')
    args = parser.parse_args()

    model_name = args.model_name if (args.model_name) else args.model_type
    train(args.model_type, model_name, args.pipeline, args.model_ckpt, args.e, args.b, args.lr, args.val_frac, args.save_best, args.clear_tb_logs,
          args.verbose)
