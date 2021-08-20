# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# References: Machine Learning A-Z: Hands-On Python & R in Data Science


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

import futils

# Command Line Arguments as follows: 
ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs


trainloader, v_loader, testloader = futils.load_data(where)


model, optimizer, criterion = futils.nn_setup(structure,dropout,hidden_layer1,lr,power)


futils.train_network(model, optimizer, criterion, epochs, 20, trainloader, power)


futils.save_checkpoint(path,structure,hidden_layer1,dropout,lr)


print("Model is successfully trained") 