from torchvision import datasets
from torch import nn, Tensor, manual_seed, load
import os.path
from digit_classifier.model.create_model import train_model, test_model 

manual_seed(42)
class MNISTModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )
    def forward(self, x: Tensor):
        return self.layer_stack(x)
    
input_shape = 784 
hidden_units = 10 
class_names = datasets.MNIST(
    root="data",
    train=True,
    download=False,
).classes
output_shape=len(class_names)
device="cpu"

def init_model():
    model_pth_path = 'model.pth'
    if os.path.exists(model_pth_path):
        print("model.pth exists, loading model state...")
        model = MNISTModel(input_shape,
            hidden_units,
            output_shape
        ).to(device)
        model.load_state_dict(load('model.pth'))
    else:
        print("model.pth does not exist, training and testing the model...")
        model = MNISTModel(input_shape,
            hidden_units,
            output_shape
        ).to(device)
        train_model(model)
        test_model(model)
    return model
