# Basic usage: python predict.py data_directory
# Uses a trained network to predict the class for an input image
# References: Machine Learning A-Z: Hands-On Python & R in Data Science

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import json
import PIL
from PIL import Image
from torch.autograd import Variable
import torchvision.models as models
import torch
from torch import nn, optim
import futils

# Command Line Arguments as follows: 

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('input_img', default='paind-project/flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='/home/workspace/paind-project/checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint



training_loader, testing_loader, validation_loader = futils.load_data()


futils.load_checkpoint(path)


with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)


probabilities = futils.predict(path_image, model, number_of_outputs, power)


labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])


i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1

print("Prediction is a Success")