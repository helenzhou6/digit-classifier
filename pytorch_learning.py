# Tutorial to learn pytorch adapted from https://www.learnpytorch.io/03_pytorch_computer_vision/
# To run file, refer to README

import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt


# 1. SETUP TRAINING AND TESTING DATA
# - MNIST is database of handwritten digits, see https://en.wikipedia.org/wiki/MNIST_database
train_data = datasets.MNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

test_data = datasets.MNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)

# -- Info on the MNIST first training and testing data

# - MNIST consists of images that are grayscale (color_channels=1) and height=28px by width=28px
image, label = train_data[0]
# print(image.shape) 

# - There are 60,000 training samples and 10,000 testing samples.
# print(len(train_data.data), len(train_data.targets), len(test_data.data), len(test_data.targets))

# - There are 10 classes - '0 - zero' ... to '9 - nine'. Therefore a multi-class classification
# class_names = train_data.classes
# print(class_names)
