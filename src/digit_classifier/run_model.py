import torch
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image

from digit_classifier.create_model import model_v1

def predict_digit_using_model(tensor_digit):
    model_v1.eval()
    with torch.no_grad():
        output = model_v1(tensor_digit)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, predicted_class = torch.max(probs, 1)
    print(f"Predicted digit: {predicted_class.item()}")
    print(f"Confidence score: {(conf.item()*100):.2f}%")

transform_to_tensor = Compose([
    Resize((28, 28)),
    ToTensor(),
])

def validate_digit_properties(tensor_digit):
    if tensor_digit.dtype is not torch.float32:
          raise Exception(f"Incorrect type - tensor digit should be torch.float32, instead was {tensor_digit.dtype}")
    if tensor_digit.shape != torch.Size([1, 28, 28]):
          raise Exception(f"Incorrect type - tensor digit should be torch.Size([1, 28, 28]), instead was {tensor_digit.shape}")
    
def process_image(uint8_img):
    #  Process image from 280 pixel x 280 pixel, 4 colour channels (3 RGB + 1 alpha) uint8 to tensor
    drawn_image = Image.fromarray(uint8_img)
    drawn_image_grey = drawn_image.convert("L") # convert to grayscale
    tensor_digit = transform_to_tensor(drawn_image_grey)
    validate_digit_properties(tensor_digit)
    return tensor_digit

def predict_digit(uint8_img):
        tensor_digit = process_image(uint8_img)
        predict_digit_using_model(tensor_digit)
