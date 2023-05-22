'''
pipelines.py

This file contains modules defining some data augmentation pipelines to use
for training. To preview a pipeline, use the following schema:

python pipelines.py <pipeline> --im_class

  + pipeline: name of the pipeline module to use
  --im_class: which image class to preview tansformations on
'''

import argparse
from utils.data import get_image, load_ppm_image, load_dataset_dict, img_tensor_to_pil
from torchvision import transforms
import torch.nn as nn
import tkinter as tk
import random
from PIL import ImageTk


def load_pipeline(pipeline):
    if pipeline == "NoiseJitterPipeline":
        return NoiseJitterPipeline()

class NoiseJitterPipeline(nn.Module):

    def __init__(self):
        super(NoiseJitterPipeline, self).__init__()
        self.pipeline = transforms.Compose([
            transforms.RandomAffine(degrees=(-5, 5), shear=(-5, 5)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=15)], p=0.05),
        ])

    def forward(self, x):
        return self.pipeline(x)

class PipelineApp:
    def __init__(self, pipeline, pipeline_name, im_class=None):
        self.pipeline = pipeline
        self.im_classes = list(load_dataset_dict().values())

        self.image_dim = 110

        self.root = tk.Tk()
        self.root.title(f"Preview Pipeline:   {pipeline_name}")
        self.canvas = tk.Canvas(self.root, width=650, height=650)
        self.canvas.pack()
        
        original_title = tk.Label(self.canvas, text="Original image", font=("Helvetica", 14, "bold"))
        original_title.grid(row=0, column=0, pady=10, columnspan=2)
        augmentations_title = tk.Label(self.canvas, text="Augmentations", font=("Helvetica", 14, "bold"))
        augmentations_title.grid(row=0, column=2, pady=10, columnspan=4)
        
        image_placeholders = []
        for i in range(5):
            placeholder = tk.Label(self.canvas, width=self.image_dim, height=self.image_dim, relief="solid", bd=1)
            placeholder.grid(row=i+1, column=1, padx=25, pady=5)
            image_placeholders.append(placeholder)
        for i in range(5):
            for j in range(4):
                placeholder = tk.Label(self.canvas, width=self.image_dim, height=self.image_dim, relief="solid", bd=1)
                placeholder.grid(row=i+1, column=j+3, padx=5, pady=5)
                image_placeholders.append(placeholder)
        self.image_placeholders = image_placeholders

        if im_class:
            image_paths = [get_image(im_class=im_class, filepath=True) for _ in range(5)]
        else:
            im_classes = random.sample(self.im_classes, 5)
            image_paths = [get_image(im_class=c, filepath=True) for c in im_classes]
        self.build_images(image_paths)

    def build_images(self, image_paths):
        for i in range(5):
            img = ImageTk.PhotoImage(load_ppm_image(image_paths[i], from_pil=True).resize([self.image_dim, self.image_dim]))
            self.image_placeholders[i].configure(image=img)
            self.image_placeholders[i].image = img
        for i in range(5):
            for j in range(4):
                idx = 5 + i*4 + j
                img_pil = img_tensor_to_pil( self.pipeline(load_ppm_image(image_paths[i])) )
                img_pil = ImageTk.PhotoImage(img_pil.resize([self.image_dim, self.image_dim]))
                self.image_placeholders[idx].configure(image=img_pil)
                self.image_placeholders[idx].image = img_pil

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preview a data augmentation pipeline")
    parser.add_argument('pipeline', help="name of the pipeline module you want to preview")
    parser.add_argument('--im_class', help="which image class to preview (optional; previews 5 random classes by defualt)")
    args = parser.parse_args()

    pipeline = load_pipeline(args.pipeline)
    PipelineApp(pipeline, args.pipeline, args.im_class).run()
