import os
from flask import Flask, request, jsonify
import tensorflow as tf
import tempfile
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelo_estima_altura_peso.keras')

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