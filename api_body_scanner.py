import os
from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub  # Para cargar el modelo de detección de personas
import tempfile
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado para predicción de altura/peso
model_path = os.path.join(os.getcwd(), 'modelo_estima_altura_peso.keras')
model = tf.keras.models.load_model(model_path)

# Cargar el modelo de detección de personas desde TensorFlow Hub
bodypix_model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Función para preprocesar la imagen
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(192, 192))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Agregar dimensión batch
    img_array /= 255.0  # Normalización
    return img_array

# Validar si el archivo subido es una imagen válida
def allowed_image(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Validar las propiedades de la imagen
def validate_image_properties(image_path):
    try:
        img = Image.open(image_path)
        width, height = img.size

        # Verificar resolución mínima (mínimo 800x1200 para capturar el cuerpo completo)
        if width < 800 or height < 1200:
            return False, "The image resolution is too low. Minimum required is 800x1200."

        # Verificar proporciones (el cuerpo debe estar completo y la imagen no debe estar distorsionada)
        aspect_ratio = height / width
        if aspect_ratio < 1.5 or aspect_ratio > 2.5:
            return False, "Invalid image aspect ratio. The expected aspect ratio should be between 1.5 and 2.5."

        return True, ""
    except Exception as e:
        return False, str(e)

# Función para detectar si la imagen contiene una persona completa (BodyPix)
def detect_full_body(image_path):
    # Abrir la imagen y redimensionar
    img = Image.open(image_path)
    img_resized = img.resize((320, 320))  # Redimensionar a 320x320 para el modelo
    img_array = np.array(img_resized).astype(np.uint8)  # Convertir a uint8
    img_array = img_array[np.newaxis, ...]  # Expandir dimensiones para batch

    # Realizar predicción con BodyPix
    bodypix_result = bodypix_model.signatures['serving_default'](tf.constant(img_array, dtype=tf.uint8))

    # Extraer cajas de detección, clases y puntuaciones
    detection_boxes = bodypix_result['detection_boxes'].numpy()  # Cajas de detección
    detection_classes = bodypix_result['detection_classes'].numpy().astype(int)  # Clases detectadas
    detection_scores = bodypix_result['detection_scores'].numpy()  # Confianza de cada detección

    # Verificar que detection_boxes tenga 3 dimensiones (batch, detecciones, coordenadas)
    if detection_boxes.ndim != 3 or detection_boxes.shape[2] != 4:
        return False

    # Iterar sobre las detecciones y verificar si alguna es una persona (clase 1)
    for i in range(detection_boxes.shape[1]):  # Iterar sobre el número de detecciones
        if detection_classes[0][i] == 1 and detection_scores[0][i] > 0.3:  # Clase 1 es "persona", confianza > 0.3
            box = detection_boxes[0][i]  # Extraer la caja i-ésima
            y_min, x_min, y_max, x_max = box  # Desempaquetar las coordenadas

            # Calcular el área de la caja de detección en relación con el tamaño de la imagen original
            box_height = y_max - y_min
            box_width = x_max - x_min
            box_area = box_width * box_height

            # Evaluar si la caja de detección cubre más del 20% de la imagen
            if box_area > 0.2:  # Si se detecta un área significativa
                return True
    return False

# Ruta para hacer predicción
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validar si hay imagen en el request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        # Validar que el archivo subido es válido
        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Validar la extensión de la imagen
        if not allowed_image(image.filename):
            return jsonify({'error': 'Invalid image format. Allowed formats are png, jpg, jpeg'}), 400

        # Guardar la imagen recibida en una ruta temporal
        filename = secure_filename(image.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
            image_path = temp_image.name
            image.save(image_path)

        # Verificar si es una imagen válida
        try:
            img = Image.open(image_path)
            img.verify()  # Esto verificará si el archivo es una imagen válida
        except (IOError, SyntaxError):
            os.remove(image_path)
            return jsonify({'error': 'Invalid image file'}), 400

        # Detectar si la imagen contiene un cuerpo completo
        if not detect_full_body(image_path):
            os.remove(image_path)
            return jsonify({'error': 'The image does not contain a full body of a person'}), 400

        # Preprocesar la imagen
        img_array = preprocess_image(image_path)

        # Hacer predicción
        prediction = model.predict(img_array)

        # Eliminar el archivo temporal
        os.remove(image_path)

        # Convertir los valores de predicción a float
        altura_predicha_cm = float(prediction[0][0])
        peso_predicho_kg = float(prediction[0][1])

        # Retornar la predicción como JSON
        return jsonify({
            'altura_cm': round(altura_predicha_cm, 2),
            'peso_kg': round(peso_predicho_kg, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ejecutar la aplicación
if __name__ == '__main__':
    # Verificar si estamos en Heroku o en local
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)