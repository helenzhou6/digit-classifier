# To run file, refer to README

import torch
from torch.utils.data import DataLoader
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

import torchmetrics

# 1. Setup training and testing data and dataloaders
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

BATCH_SIZE = 32
train_dataloader = DataLoader(train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# 2. Create Model v1 with non-linearity and linear layers
torch.manual_seed(42)
class MNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)

input_shape = 784 
hidden_units = 10 
class_names = train_data.classes
output_shape=len(class_names)
device="cpu"
model_v1 = MNISTModelV1(input_shape,
    hidden_units,
    output_shape=len(class_names)
).to(device)

# 3. Train model v1
accuracy_fn = torchmetrics.Accuracy(task = 'multiclass', num_classes=len(class_names)).to(device)
loss_fn = nn.CrossEntropyLoss()
model_v1_optimizer = torch.optim.SGD(params=model_v1.parameters(), lr=0.1)

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn):
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        model.train()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss 
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1)) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

epochs = 3 
for epoch in range(epochs):
    print(f"Training: Epoch {epoch + 1} out of {epochs}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_v1, 
        loss_fn=loss_fn,
        optimizer=model_v1_optimizer,
        accuracy_fn=accuracy_fn
    )

# 4. Evaluade the model and get accuracy metrics
def eval_model(data_loader: torch.utils.data.DataLoader,
               model: torch.nn.Module, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_loss": f"{loss.item():.5f}%",
            "model_acc": f"{acc:.2f}%"}

model_v1_results = eval_model(data_loader=test_dataloader, model=model_v1,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn
)
print(model_v1_results)