import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image




class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Input size: [1, 300, 300]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        # Output size: [32, 150, 150]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output size: [32, 75, 75]

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        # Output size: [64, 38, 38]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output size: [64, 19, 19]

        self.flatten = nn.Flatten()
        # Output size: [64 * 19 * 19]

        self.fc1 = nn.Linear(64 * 19 * 19, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Load the model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model('cnn_model.pth')

# Streamlit interface
st.title("Manufacturing defect classification:")

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

def predict(image, model):
    # Preprocess the image to be compatible with the model
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (300, 300))  # Resize to the size required by the model
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  
    image = np.expand_dims(image, axis=0) 
    image = torch.from_numpy(image).float()

    with torch.no_grad():
        output = model(image)
        prediction = torch.round(output)  
        return prediction.item()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(image, model)
    st.write(f"The casting is classified as: {'defecting' if label == 1 else 'OK'}")
