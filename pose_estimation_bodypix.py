import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Cargar el modelo BodyPix desde TensorFlow Hub para Python
model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")

# Función para segmentar el cuerpo
def segment_body(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Redimensionar la imagen para que coincida con el modelo
    input_image = tf.image.resize(img_rgb, [224, 224])
    input_image = tf.cast(input_image, dtype=tf.float32) / 255.0
    input_image = tf.expand_dims(input_image, axis=0)

    # Ejecutar el modelo
    result = model(input_image)

    # Aquí se ajusta el acceso a la salida del modelo
    segmentation_mask = result[0].numpy()  # Extrayendo la segmentación de la primera capa de salida
    return segmentation_mask, img.shape

# Función para estimar la altura en píxeles a partir de la máscara
def estimate_height_in_pixels(mask):
    indices = np.where(mask > 0.5)  # Máscara binaria
    if len(indices[0]) == 0:
        return None  # Si no se detecta la persona correctamente
    head_y = np.min(indices[0])  # Coordenada Y más alta (la cabeza)
    feet_y = np.max(indices[0])  # Coordenada Y más baja (los pies)
    height_in_pixels = feet_y - head_y
    return height_in_pixels

# Función para convertir la altura en píxeles a altura real
def convert_pixels_to_height(height_in_pixels, image_height, camera_distance_m=2):
    if height_in_pixels is None:
        return None  # Si no se detectó una persona
    pixels_per_meter = image_height / camera_distance_m  # Relación píxeles por metro
    height_in_meters = height_in_pixels / pixels_per_meter
    return height_in_meters * 100  # Convertir metros a centímetros

# Estimación de peso usando el IMC
def estimate_weight(height_cm, imc=22):
    height_m = height_cm / 100
    return imc * (height_m ** 2)

# Procesar una imagen y estimar altura y peso
def process_image(image_path):
    mask, image_shape = segment_body(image_path)
    height_in_pixels = estimate_height_in_pixels(mask)

    if height_in_pixels is None:
        print("No se pudo detectar una persona en la imagen.")
        return None, None

    height_cm = convert_pixels_to_height(height_in_pixels, image_shape[0])

    if height_cm is None:
        return None, None

    estimated_weight_kg = estimate_weight(height_cm)

    return height_cm, estimated_weight_kg

# Preprocesar y mostrar la máscara segmentada
def visualize_segmentation(image_path):
    mask, img_shape = segment_body(image_path)
    
    if mask is not None:
        plt.imshow(mask)
        plt.title("Segmentación del cuerpo")
        plt.show()
    else:
        print("No se pudo detectar una persona en la imagen.")

# Ejemplo de uso
image_path = r"C:\Users\svega\OneDrive\Pictures\Saved Pictures\depositphotos_687682370-stock-photo-full-body-front-view-young.jpg"
height_cm, weight_kg = process_image(image_path)

if height_cm and weight_kg:
    print(f"Altura estimada: {height_cm:.2f} cm")
    print(f"Peso estimado: {weight_kg:.2f} kg")

# Mostrar la segmentación para asegurarse de que el modelo funciona correctamente
visualize_segmentation(image_path)