from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# Asegúrate de que el directorio de subidas exista
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cargar el modelo preentrenado
model = load_model('model/modelo_clasificacion.h5')

# Definir las clases
CLASSES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

# Ruta principal para mostrar el formulario
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para cargar y clasificar la imagen
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            # Clasificar la imagen
            classification = classify_image(filepath)
            # Eliminar el archivo subido para evitar acumulación
            os.remove(filepath)
            return jsonify({'classification': classification['class'], 'confidence': classification['confidence']})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

# Función para clasificar la imagen
def classify_image(filepath):
    # Cargar y preprocesar la imagen
    img = image.load_img(filepath, target_size=(128, 128))  # Ajusta el tamaño según tu modelo
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
    
    # Realizar la predicción
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = float(pred[0][class_idx])  # Convertir a tipo float compatible con JSON
    confidence_percentage = round(confidence * 100, 2)  # Convertir a porcentaje
    return {'class': CLASSES[class_idx], 'confidence': confidence_percentage}


if __name__ == '__main__':
    app.run(debug=True)
