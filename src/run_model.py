import torch
import matplotlib.pyplot as plt

from create_model import model_v1, test_data, class_names

# 5. Test out with an input
image, label = test_data[201]
plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label]);
plt.axis("Off");
plt.show()

def predict_digit(image):
    model_v1.eval()
    with torch.no_grad():
        output = model_v1(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, predicted_class = torch.max(probs, 1)
    print(f"Predicted digit: {predicted_class.item()}")
    print(f"Confidence score: {(conf.item()*100):.2f}%")

predict_digit(image)
