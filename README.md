# TrafficSignClassifier
Pytorch implementation of a machine learning project for classifying images of traffic signs from the German Traffic Sign
dataset (GTSRB).

![Screenshot 2023-05-22 150059](https://github.com/lliv12/TrafficSignClassifier/blob/master/assets/Screenshot%202023-05-22%20150059.png)

### Dataset
The GTSRB dataset consists of over 50,000 labeled images of traffic signs, belonging to 43 different classes.
The **Test** directory consists of ~12,000 unlabled .ppm images. The **Train** directory contains 43 folders
(corresponding to each class), ranging from '00000' to '00042'. Within each folder there are several houndred images
formatted as '000XX_000ii.ppm' where XX is the image instance, and ii is the resolution (00: lowest, 29: highest).

[Link to the dataset](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html)

### Model
The pretrained model is a basic CNN (refer to RegularClassifier in models.py for the exact architecture). This model
achieves ~87% accuracy on the test set after 2 epochs of training. Data augmentation was used; train images were
augmented with random color jitter and gaussian noise to help account for varying resolution of images.

# Getting Started
#### Step 1: Setting up the Environment (Conda)
<pre>conda env create -f environment.yml
conda activate gtsrb_env</pre>
Use Anaconda to install environment dependencies (or refer to requirements.txt for required dependencies).
Run **conda activate gtsrb_env** everytime you want to run the code, as this will activate the environment
with required dependencies.
#### Step 2: Download the dataset
<pre>python -m utils.download_data</pre>
#### Step 3: Download the pretrained model
<pre>python -m utils.download_models</pre>
The pretrained model will be saved in **saved_models/regular_classifier_best.pt**

# Using the Application (inference)
<pre>python inference.py <model> --model_ckpt</pre>
* **model**:       the type of model to inference (Ex: 'RegularClassifier')
* **model_ckpt**:  the model checkpoint to load (Ex: 'regular_classifier_best.pt')

#### Example:
<pre>python inference.py RegularClassifier --model_ckpt regular_classifier_best.pt</pre>

The application can be started by running **inference.py**. A window will pull up with an image of a
stop sign at the top, and a bar graph on the lower-half. The displayed image is the current selected image from
the dataset (a stop sign by default). To change this, click on either of the two buttons at the bottom of the
window. The first button lets you select an image class from a dropdown menu. The second button lets you select
an image from the dataset directly.

Below the image is a bar graph showing the model's predictions. On the y-axis is the top-5 classes listed from
most likely (top) to least likely (bottom). On the x-axis is the model's confidence for each classification,
ranging from 0.0 to 1.0. The bar that is green is for the correct class.

<p align="center">
  <img src="https://github.com/lliv12/TrafficSignClassifier/blob/master/assets/Screenshot%202023-05-22%20150419.png" alt="image" style="width:400px">
</p>

# Training
To begin training a model, run the following script according to this schema:
<pre>python train.py <model_type> --model_ckpt --model_name --pipeline --e --b --val_frac --save_best --clear_tb_logs --verbose</pre>
* **model_type**:    the type of model to train (Ex: 'ResNet50')
* **model_ckpt**:    (optional) start from a pretrained model checkpoint
* **model_name**:    name of the model. Will default to <model_type>
* **pipeline**:      (optional) which data augmentation pipeline to use (refer to pipelines.py)
* **e**:             how many epochs to run training for
* **b**:             the batch size to use for training
* **val_frac**:      the fraction of the training data to use for validation (Ex: 0.2)
* **save_best**:     (default: True) save the best model checkpoint rather than the last model
* **clear_tb_logs**: (default: True) clear tensorboard logs for this model before starting the run
* **verbose**:       (default: True) log model performance to the console

#### Example:&nbsp;&nbsp;&nbsp;(Train RegularClassifier with data augmentation for 2 epochs)
<pre>python train.py RegularClassifier --model_name regular_classifier_2 --pipeline NoiseJitterPipeline --e 2</pre>

### Tensorboard
When running train.py, a TensorBoard session can be launched in one of the following ways:

(With Tensorboard extension in VS Code):
1) Ctrl + Shift + P  to view commands
2) Select "Launch Tensorboard" >> "Use current working directory" (or select folder where logs is contained)

(Without Tensorboard Extension):
1) Open a separate terminal and run "tensorboard --logdir=~/TrafficSignClassifier/logs"
2) Go to the link provided in your web browser (Ex:  "http://localhost:6006/")

NOTE:  Refresh a couple of times until the graphs show up

# Testing:
<pre>python test.py <model> --model_ckpt --batch_size</pre>
* **model**:      name of the model architecture (Ex: 'RegularClassifier')
* **model_ckpt**: name of the model checkpoint / absolute path to the checkpoint
* **batch_size**: batch size to use for the test set loader

#### Example:
<pre>python test.py RegularClassifier --model_ckpt regular_classifier_best.pt</pre>
Run the script **test.py** to begin evaluating a given model on the test set.

# Other:
### Preview Data
<pre>python utils/data.py --filepath --im_class --instance --resolution</pre>
* **filepath**:   filepath to the image
* **im_class**:   which image class you want to visualualize (Ex: 'stop_sign')
* **instance**:   which image instance you want to visualize
* **resolution**: which instance resolution you want to visualualize (0-29)

Display an image instance from the dataset.

### Visualize Pipelines
<pre>python pipelines.py <pipeline> --im_class</pre>
* **pipeline**: name of the pipeline module to use (Ex: 'NoiseJitterPipeline')
* **im_class**: which image class to preview tansformations on (Ex: 'stop_sign')

#### Example:
<pre>python pipelines.py NoiseJitterPipeline</pre>

Preview an image pipeline to get an idea of what images look like after data augmentation. If no image class is
specified, the script will display augmentations on 5 random image classes by default.

<p align="center">
  <img src="https://github.com/lliv12/TrafficSignClassifier/blob/master/assets/Screenshot%202023-05-22%20154421.png" alt="image" style="width:430px">
</p>
