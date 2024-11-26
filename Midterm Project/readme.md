# Heart Disease Prediction App

## Problem Description

Heart diseases are one of the leading causes of death worldwide. Early risk detection can make the difference between life and death. This project uses a machine learning model to predict the probability of a patient having heart disease.

### How It Works

The model is trained on a patient dataset including variables such as age, gender, cholesterol, and other risk factors. Once trained, the model can predict whether a patient tests positive or negative for heart disease based on their individual characteristics.

## Key Features

- **Decision Tree Model**: Fast and easy to interpret
- **Standard Scaler**: For normalizing input features
- **User-Friendly Web Interface**: Built with Flask and HTML, allowing users to input data and receive real-time predictions
- **Docker Container**: For easy deployment across different environments

## Prerequisites

- Python 3.8+
- Docker (optional)
- Pipenv

## How to Run the Project

### Option 1: Using Docker

1. **Clone the Repository**

   ```bash
   git clone https://github.com/JorgeSandon/ML-ZoomCamp-Cohorts-2024.git
   cd ML-ZoomCamp-Cohorts-2024/Midterm\ Project
   ```

2. **Build Docker Image**

   ```bash
   docker build -t flask-heart-disease-app .
   ```

3. **Run the Container**

   ```bash
   docker run -p 5000:5000 flask-heart-disease-app
   ```

   The application will be available at http://localhost:5000.

### Option 2: Run Locally

1. **Install Dependencies**

   ```bash
   pip install pipenv
   pipenv install
   ```

2. **Run the Application**

   ```bash
   pipenv run python predict.py
   ```

   The application will be available at http://localhost:5000.

## Model Training

1. **Update Data**

   Replace the `data/heart.csv` file with a more recent dataset if necessary.

2. **Train Model**

   ```bash
   python train.py
   ```

   This will generate a new model and scaler that will be saved in the `model/` folder.

## Project Structure

```
project/
├── data/
│   ├── data.py       # Data loading and preprocessing code
│   └── heart.csv     # Dataset
├── model/
│   ├── dtc_model.pkl # Serialized model
│   └── scaler.pkl    # Serialized scaler
├── templates/
│   ├── index.html    # Main form page
│   └── result.html   # Results display page
├── Dockerfile        # Docker image build file
├── predict.py        # Flask prediction application
├── train.py          # Model training and saving script
├── Pipfile           # Project dependencies
├── Pipfile.lock      # Dependency lockfile
└── README.md         # Project documentation
```
