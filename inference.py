'''
inference.py

Module for inferencing with trained model. Will launch the application with interface.
Use this schema to run the application:

python inference.py <model> --model_ckpt

  + model:      name of the model architecture (module class from models.py)
  --model_ckpt: name of the model checkpoint / absolute path to the checkpoint
'''

import tkinter as tk
from tkinter.filedialog import askopenfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk
from torch.nn.functional import softmax
from utils.data import get_image, load_ppm_image, load_dataset_dict
from models import load_model
import torch
import argparse

DATA_DIR = 'data'

class TSC_App:
    def __init__(self, model_name, model_ckpt=None):
        self.model = load_model(model_name, model_ckpt)
        self.model.eval()
        self.dataset_dict = load_dataset_dict()
        self.selection = 'stop_sign'

        # Root & canvas
        self.root = tk.Tk()
        self.root.title("Traffic Sign Classifier")
        self.canvas = tk.Canvas(self.root, width=650, height=650)
        self.canvas.grid(columnspan=4, rowspan=3)
        self.plot_figure = plt.figure(figsize=(6,3), dpi=100, tight_layout={'pad':1.5, 'h_pad':-17})

        # load default image
        self.image_dim = (300, 300)
        self.current_image_path = get_image(filepath=True, dataset='Train', im_class=self.selection, instance=8, resolution=0)
        self.current_gui_image = ImageTk.PhotoImage( Image.open(self.current_image_path).resize(self.image_dim) )

        # Image Widgit
        self.image_label = tk.Label(image=self.current_gui_image)
        self.image_label.Image = self.current_gui_image
        self.image_label.grid(columnspan=2, column=1, row=0)

        # Select Class Button
        self.select_class_button_selection = tk.StringVar()
        self.select_class_button_selection.set(self.selection)
        self.select_class_button = tk.OptionMenu(self.root, self.select_class_button_selection, *self.dataset_dict.values(), command=self.select_class_button_f)
        self.select_class_button.config(bg='#e0edea', height=3, width=16)
        self.select_class_button.grid(column=1, row=3)

        # Select From Dataset Button
        self.select_dataset_file_button = tk.Button(self.root, text="Select From Dataset", command=self.select_from_dataset_button_f, bg='#e0edea', fg='black', height=3, width=16)
        self.select_dataset_file_button.grid(column=2, row=3)

        self.__update_plot()

    def run(self):
        self.root.mainloop()

    def __update_image_selection(self, directory):
        self.current_image_path = directory
        self.current_gui_image = ImageTk.PhotoImage( Image.open(self.current_image_path).resize(self.image_dim) )
        self.image_label.configure(image=self.current_gui_image)
        self.image_label.Image = self.current_gui_image

    def __update_plot(self):
        predictions = self.predict_on_image(self.model, directory=self.current_image_path)
        sorted_indices = torch.argsort(predictions).squeeze()
        pred_sorted = predictions[0][sorted_indices].squeeze()

        top5_classes = sorted_indices[-5:]
        labels = [self.dataset_dict[ str(top5_classes[i].item()).zfill(5) ] for i in range(5)]
        values = softmax(pred_sorted[-5:], dim=-1).detach().numpy()
        colors = ['green' if label == self.selection else 'tab:blue' for label in labels]

        plt.clf()
        plt.subplot(2, 1, 2)
        plt.barh(labels, values, color=colors)
        plt.xlabel('Confidence')
 
        plot_canvas = FigureCanvasTkAgg(self.plot_figure, master=self.root)
        plot_canvas.draw()
        plot_canvas.get_tk_widget().grid(columnspan=2, column=1, row=2)

        toolbar = NavigationToolbar2Tk(plot_canvas, self.root, pack_toolbar=False)
        toolbar.update()

    # Button functionality
    def select_class_button_f(self, selection):
        self.selection = selection
        self.__update_image_selection(get_image(filepath=True, im_class=selection))
        self.__update_plot()

    def select_from_dataset_button_f(self):
        im_file = askopenfile(parent=self.root, initialdir=DATA_DIR, mode='r', title="Select an Image File", filetype=[("ppm file", '*.ppm')])
        self.__update_image_selection(im_file.name)
        self.__update_plot()

    def predict_on_image(self, model, directory):
        return model( load_ppm_image(directory).unsqueeze(0) )


if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='the name of the model to load and do inference on.')
    parser.add_argument('--model_ckpt', help='the model checkpoint to load (optional)')
    args = parser.parse_args()

    TSC_App(args.model, args.model_ckpt).run()
