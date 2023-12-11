# Casting Product Classification

## Project Overview
This repository contains a machine learning project focused on the binary classification of casting products into defective and non-defective categories. The project is based on a [real-life industrial dataset from Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/data), and employs Convolutional Neural Networks (CNN) using PyTorch.

## Features
- **Model Training**: Code for training a CNN and OpenCV model in PyTorch.
- **Inference Pipeline**: A Streamlit application for easy model inference.
- **Visualization**: Code for plotting confusion matrices and displaying false positives and false negatives.

## Installation
To run this project, first clone the repository:
```bash
git clone https://github.com/[your-username]/[your-repo-name].git
cd [your-repo-name]
python3 -m venv myenv
source myenv/bin/activate
streamlit run inference.py
pip install -r requirements.txt
streamlit run app.py
.
├── cnn_model.py       # CNN model definition
├── experiment.ipynb    #Experimentational script
├── inference.py       # Inference code for model evaluation
├── cnn_model.pth      # Trained model file
├── requirements.txt   # Required Python libraries
└── README.md          # This file

