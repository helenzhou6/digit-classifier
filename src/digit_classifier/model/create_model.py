import torch
from torch.utils.data import DataLoader
from torch import nn

from torchvision import datasets
from torchvision.transforms import ToTensor

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
class_names = train_data.classes
device="cpu"


# 2. See model.py that creates the model with non-linearity and linear layers
# 3. Train model
def _train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn: torchmetrics.classification.Accuracy):
    train_loss, train_acc = 0, 0
    for _, (X, y) in enumerate(data_loader):
        model.train()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss 
        train_acc += accuracy_fn(y_pred.argmax(dim=1), y) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {(train_acc*100):.2f}%")

accuracy_fn = torchmetrics.Accuracy(task = 'multiclass', num_classes=len(class_names)).to(device)
loss_fn = nn.CrossEntropyLoss()

def train_model(model, epochs = 3):
    model_optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    for epoch in range(epochs):
        print(f"---------\nTraining: Epoch {epoch + 1} out of {epochs}")
        _train_step(data_loader=train_dataloader, 
            model=model, 
            loss_fn=loss_fn,
            optimizer=model_optimizer,
            accuracy_fn=accuracy_fn
        )

# 4. Evaluade the model and get accuracy metrics
def _eval_model(data_loader: torch.utils.data.DataLoader,
               model: torch.nn.Module, 
               loss_fn: torch.nn.Module, 
               accuracy_fn: torchmetrics.classification.Accuracy):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_pred.argmax(dim=1), y)
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_loss": loss.item(),
            "model_acc": acc*100}

def test_model(model):
    model_v1_results = _eval_model(data_loader=test_dataloader, model=model,
        loss_fn=loss_fn, accuracy_fn=accuracy_fn
    )
    model_results_loss = model_v1_results['model_loss']
    model_results_accuracy = model_v1_results['model_acc'].item()
    if model_v1_results['model_loss'] < 0.5 and model_v1_results['model_acc'].item() > 90:
        print(f"Model test results: loss={model_results_loss:.5f}%, accuracy={model_results_accuracy:.2f}%")
        print("PASSED - model is usable")
        # Save state of the model into .pth file 
        torch.save(model.state_dict(), 'model.pth')
    else:
        raise Exception(f"Machine learning model not usable - since model_loss was > 0.5 at {model_results_loss} and accuracy was < 90 at {model_results_accuracy}")
