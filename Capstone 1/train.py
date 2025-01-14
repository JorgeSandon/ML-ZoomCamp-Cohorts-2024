import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_data_generators(train_dir, val_dir, test_dir):
    """
    Crea generadores de datos para las carpetas de entrenamiento, validación y prueba.
    """
    logging.info("Creando generadores de datos...")
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )
    
    val_data = datagen.flow_from_directory(
        val_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )
    
    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )
    
    logging.info("Generadores de datos creados con éxito.")
    return train_data, val_data, test_data

def build_model(input_shape=(128, 128, 3)):
    """
    Construye y compila el modelo de red neuronal.
    """
    logging.info("Construyendo el modelo...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    logging.info("Modelo construido y compilado.")
    return model

def plot_training(history):
    """
    Genera gráficas de precisión y pérdida a lo largo de las épocas.
    """
    logging.info("Generando gráficas de entrenamiento...")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    # Gráfica de precisión
    plt.plot(epochs, acc, 'g', label='Entrenamiento')
    plt.plot(epochs, val_acc, 'b', label='Validación')
    plt.title('Precisión del Modelo a lo Largo de las Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid()
    plt.figure()

    # Gráfica de pérdida
    plt.plot(epochs, loss, 'g', label='Entrenamiento')
    plt.plot(epochs, val_loss, 'b', label='Validación')
    plt.title('Pérdida del Modelo a lo Largo de las Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid()
    plt.figure()

    # Mostrar gráficas
    plt.show()

def main():
    # Directorios de datos
    train_dir = "data/train"
    val_dir = "data/val"
    test_dir = "data/test"

    # Crear generadores de datos
    train_data, val_data, test_data = create_data_generators(train_dir, val_dir, test_dir)

    # Construir y compilar el modelo
    model = build_model()

    # Entrenar el modelo
    logging.info("Iniciando el entrenamiento del modelo...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=7,
        verbose=1
    )
    logging.info("Entrenamiento finalizado.")

    # Evaluar el modelo
    logging.info("Evaluando el modelo en el conjunto de prueba...")
    test_loss, test_accuracy = model.evaluate(test_data)
    logging.info(f"Precisión en el conjunto de prueba: {test_accuracy:.2f}")

    # Guardar el modelo
    model_path = "model/modelo_clasificacion.h5"
    model.save(model_path)
    logging.info(f"Modelo guardado en: {model_path}")

    # Generar gráficas
    plot_training(history)

if __name__ == "__main__":
    main()
