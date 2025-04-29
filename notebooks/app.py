import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm

# Define class names
class_names = ['no-watermark', 'watermark']

# Load model
@st.cache_resource
def load_model():
    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=2)
    model.load_state_dict(torch.load('logoconvnext_best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image transformation
input_size = 256
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(image):
    image = transform(image).unsqueeze(0)  
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# Streamlit UI
st.title("Watermark Detection using ConvNeXt")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        prediction = predict_image(image)
        st.success(f"Prediction: **{prediction}**")
