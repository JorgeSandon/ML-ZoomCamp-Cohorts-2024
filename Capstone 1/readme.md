# Chest X-Ray Classification Application

This repository contains a Flask-based web application for classifying chest X-ray images into four categories: NORMAL, PNEUMONIA, COVID-19, and TUBERCULOSIS. The application leverages a pre-trained convolutional neural network (CNN) model to make predictions.

---

## Problem Description

Accurate and timely diagnosis of respiratory diseases such as pneumonia, COVID-19, and tuberculosis is critical for effective treatment and patient management. Chest X-rays are a widely used diagnostic tool, but interpreting them requires expertise. This application aims to assist medical professionals by providing a machine-learning-based solution for classifying chest X-ray images.

---

## Features

- **Image Upload:** Upload chest X-ray images for classification.
- **Real-Time Predictions:** Get predictions from a pre-trained model.
- **Interactive Web Interface:** Simple and intuitive interface built with Flask.

---

## Setup Instructions

### Prerequisites

- Python 3.12
- Pipenv for virtual environment management
- Docker (if running the application in a container)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JorgeSandon/ML-ZoomCamp-Cohorts-2024
   cd ML-ZoomCamp-Cohorts-2024
   ```

2. **Install dependencies:**
   - Create and activate a virtual environment using Pipenv:
     ```bash
     pip install pipenv
     pipenv install
     ```
   - Activate the environment:
     ```bash
     pipenv shell
     ```

3. **Run the application locally:**
   ```bash
   python app.py
   ```

4. **Access the application:**
   Open a browser and go to `http://localhost:5000`.

---

## Docker Instructions

To containerize and run the application:

1. **Build the Docker image:**
   ```bash
   docker build -t chest-xray .
   ```

2. **Run the Docker container:**
   ```bash
   docker run -p 5000:5000 chest-xray
   ```

3. **Access the application:**
   Open a browser and go to `http://localhost:5000`.

---

## Directory Structure

```
data/
    ├── test/                # Images for testing
    ├── train/               # Images for training
    ├── val/                 # Images for validation
    └── data.py              # Script for downloading and preparing the data
model/
    └── modelo_clasificacion.h5  # Pre-trained model
notebooks/ 
    └── notebook.ipynb       # Jupyter notebook for exploratory data analysis
templates/
    └── index.html           # HTML template for the web interface
app.py                       # Main Flask application
train.py                     # Script for training the model
Pipfile                      # Pipenv configuration file
Pipfile.lock                 # Locked Pipenv dependencies
requirements.txt             # Python dependencies for Docker
Dockerfile                   # Docker configuration
service.yaml                 # Kubernetes Service configuration
deployment.yaml              # Kubernetes Deployment configuration
```

---

## Kubernetes Deployment

1. **Create the Kubernetes cluster:**
   ```bash
   kind create cluster
   ```

2. **Deploy the application:**
   ```bash
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   ```

3. **Access the application:**
   Get the cluster IP address and port:
   ```bash
   kubectl get service
   ```
   Open a browser and go to the listed external IP and port.

---

## Notes

- Ensure the dataset is not included in the Docker image. Use `data.py` to download and prepare the data as needed.
- The `model/modelo_clasificacion.h5` file must be included for predictions.

---

## Demo

A short demonstration of the application in action:

<video src="video\test video.mp4" controls width="100%"></video>

---

## Contributors

- Jorge Elias Sandon Calderin

---

## License

This project is licensed under the MIT License.
