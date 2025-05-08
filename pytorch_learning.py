# Tutorial to learn pytorch adapted from https://www.learnpytorch.io/03_pytorch_computer_vision/

# Import PyTorch
import torch
from torch import nn

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt


# Setup training data. MNIST is database of handwritten digits, see https://en.wikipedia.org/wiki/MNIST_database
train_data = datasets.MNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.MNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)

# See first training sample
image, label = train_data[0]
# based on image.shape, grayscale (color_channels=1) and 28px by 28px
print(image.shape)