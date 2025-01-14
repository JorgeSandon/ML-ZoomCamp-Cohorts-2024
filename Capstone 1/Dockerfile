# Usamos una imagen base con Python 3.10
FROM python:3.10-slim

# Seteamos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos el archivo requirements.txt al contenedor
COPY requirements.txt /app/

# Instalamos las dependencias de la aplicación desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código de la aplicación al contenedor
COPY . /app/

# Exponemos el puerto 5000, que es el predeterminado para Flask
EXPOSE 5000

# Definimos el comando para ejecutar la aplicación Flask
CMD ["python", "app.py"]

