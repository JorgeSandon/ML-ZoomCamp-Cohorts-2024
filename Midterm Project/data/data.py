import subprocess
import zipfile
import os

# Download the dataset to 'data/' (force download if it already exists)
subprocess.run(["kaggle", "datasets", "download", "johnsmith88/heart-disease-dataset", "-p", "data/", "--force"])

# Path of the downloaded ZIP file
zip_path = "data/heart-disease-dataset.zip"
extract_path = "data/"

# Check if the file exists before trying to unzip
if os.path.exists(zip_path):
    # Unzip the file using zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("File successfully unzipped to 'data/'")
else:
    print("ZIP file not found for extraction.")




