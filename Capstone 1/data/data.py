import subprocess
import zipfile
import os


# Download the dataset to 'data/' (force download if it already exists)
subprocess.run(["kaggle", "datasets", "download", "jtiptj/chest-xray-pneumoniacovid19tuberculosis", "-p", "data/", "--force"])

# Path of the downloaded ZIP file
zip_path = "data/chest-xray-pneumoniacovid19tuberculosis.zip"
extract_path = "data/"

# Comprobar si el archivo existe antes de intentar descomprimirlo
if os.path.exists(zip_path):
    # Descomprimir el archivo usando zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Archivo descomprimido correctamente en 'data/'")
    
    # Eliminar el archivo ZIP después de descomprimirlo
    os.remove(zip_path)
    print(f"Archivo ZIP '{zip_path}' eliminado correctamente.")
else:
    print("No se encontró el archivo ZIP para descomprimir.")




