import torch
from torchvision import transforms
from PIL import Image
import os
import timm

# Define the class names
class_names = ['no-watermark', 'watermark']  

model = timm.create_model('convnext_tiny', pretrained=True, num_classes=2)
model.load_state_dict(torch.load('logoconvnext_best_model.pth'))
model.eval()

input_size = 256
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    prediction = class_names[predicted.item()]  # Map the class index to the class name
    print(f"The image at {image_path} is predicted to be: {prediction}")

if __name__ == "__main__":
    image_path = input("Enter the path to the image: ")
    
    if not os.path.exists(image_path):
        print(f"The path {image_path} does not exist. Please provide a valid path.")
    else:
        predict_image(image_path)
