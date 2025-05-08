# Tutorial to learn pytorch adapted from https://www.learnpytorch.io/03_pytorch_computer_vision/
# To run file, refer to README

import torch
from torch.utils.data import DataLoader
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

import torchmetrics
from tqdm.auto import tqdm

# 1. SETUP TRAINING AND TESTING DATA
# - MNIST is database of handwritten digits, see https://en.wikipedia.org/wiki/MNIST_database
train_data = datasets.MNIST(
    root="data", # downloads to local data folder
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

# - MNIST consists of images that are grayscale (color_channels=1) and height=28px by width=28px (see image.shape)
image, label = train_data[0]

# - There are 60,000 training samples and 10,000 testing samples.
# print(len(train_data.data), len(train_data.targets), len(test_data.data), len(test_data.targets))

# - There are 10 classes - '0 - zero' ... to '9 - nine'. Therefore a multi-class classification
class_names = train_data.classes

# 2. VISUALISE THE DATA
# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(class_names[label]);
# plt.axis("Off");
# plt.show()

# 3. Prepare DataLoader
# Turn datasets into iterables (batches of 32)
BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)
# print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
# Length of train dataloader: 1875 batches of 32
# print(f"L÷ength of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
# Length of test dataloader: 313 batches of 32

# Inside the training dataloader
# train_features_batch, train_labels_batch = next(iter(train_dataloader))
# print(train_features_batch.shape, train_labels_batch.shape)

# 4. Build a baseline model (model 0 - one of the simplest models, used as a starting point)
# Create a flatten layer - compresses the dimensions of a tensor into a single feature vector (height*width)
# because nn.Linear() layers like inputs to be in the form of feature vectors

# EXAMPLE: Get a single sample and flatten the sample
# flatten_model = nn.Flatten() # all nn modules function as a model (can do a forward pass)
# x = train_features_batch[0]
# output = flatten_model(x) # perform forward pass
# print(f"Shape before flattening: {x.shape} -> [color_channels, height, width] VS after flattening: {output.shape} -> [color_channels, height*width]")

# 4a. Baseline model class 
class MNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = number of features in a data sample
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)
    
# 4b. Instantiate/setup a model
# number of features in the model, for here it's one for every pixel (28 pixels high x 28 pixels wide = 784 features).
input_shape = 784 
# number of units/neurons in the hidden layer, can be whatever you want but to keep the model small start with 10.
hidden_units = 10 
# since we're working with a multi-class classification problem, we need an output neuron per class in our dataset.
output_shape=len(class_names)

torch.manual_seed(42) # sets the seed for generating random numbers to ensure random numbers are generated the same each time run the code
device="cpu"

model_0 = MNISTModelV0(input_shape,
    hidden_units,
    output_shape
)
model_0.to(device)

# 6. Model v1 with non-linearity and linear layers
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # flatten inputs into single vector
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(), # non-linear functions added in between each linear layer
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)

model_1 = FashionMNISTModelV1(input_shape=784, # number of input features
    hidden_units=10,
    output_shape=len(class_names) # number of output classes desired
).to(device)

# 5. Set up loss, optimizer and evaluation metrics
# Set up accuracy metric
accuracy_fn = torchmetrics.Accuracy(task = 'multiclass', num_classes=len(class_names)).to(device)
# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss() # this is also called "criterion"/"cost function" in some places
model_0_optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)
model_1_optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

# 6. Create training & testing loop. Train model 0 on batches of data
epochs = 3 # number of epochs (i.e. complete pass through the complete dataset. set as small for faster training times)
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    ### Training
    train_loss = 0
    # Add a loop to loop through training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train() 
        # 1. Forward pass (perform training steps)
        y_pred = model_0(X)
        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulatively add up the loss per epoch 
        # 3. Optimizer zero grad
        model_0_optimizer.zero_grad()
        # 4. Loss backward
        loss.backward()
        # 5. Optimizer step
        model_0_optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")
    # Divide total train loss by length of train dataloader (average train loss per batch per epoch)
    train_loss /= len(train_dataloader)
    
    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy 
    test_loss, test_acc = 0, 0 
    model_0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # 1. Forward pass
            test_pred = model_0(X)
            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch
            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy_fn(y, test_pred.argmax(dim=1))
        
        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dataloader)

    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

# 7. Make predictions and get Model 0 results
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader.
    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": f"{loss.item():.5f}%",
            "model_acc": f"{acc:.2f}%"}

# Calculate model 0 results on test dataset
model_0_results = eval_model(model=model_0, data_loader=test_dataloader,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn
)
print(model_0_results)