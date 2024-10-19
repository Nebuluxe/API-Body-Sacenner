import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Cargar el CSV
data = pd.read_csv(r'C:\\Python Projects\\Output_data.csv')

# Función para convertir alturas en pies y pulgadas a centímetros
def convert_height(height_str):
    feet, inches = height_str.split("'")
    inches = inches.replace('"', '').strip()  # Limpiar las comillas
    total_inches = int(feet.strip()) * 12 + int(inches)
    height_cm = total_inches * 2.54  # Convertir a centímetros
    return height_cm

# Función para convertir pesos en libras a kilogramos
def convert_weight(weight_str):
    weight_lbs = int(weight_str.replace('lbs.', '').strip())
    weight_kg = weight_lbs * 0.453592  # Convertir a kilogramos
    return weight_kg

# Procesar las columnas de altura y peso
data['Height_cm'] = data['Height & Weight'].apply(lambda x: convert_height(x.split()[0] + " " + x.split()[1]))
data['Weight_kg'] = data['Height & Weight'].apply(lambda x: convert_weight(x.split()[2] + " " + x.split()[3]))

# Asegurar que las rutas de las imágenes sean correctas
data['Filepath'] = data['Filename'].apply(lambda x: os.path.join(r'C:\\Python Projects\\TestImages', x))

# Verificar si una imagen es válida
def verify_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()  # Verifica si es una imagen válida
        return True
    except (IOError, SyntaxError) as e:
        print(f"Imagen no válida: {filepath}")
        return False

# Filtrar las imágenes válidas en el dataframe
data['Valid'] = data['Filepath'].apply(verify_image)
valid_data = data[data['Valid'] == True]  # Solo mantener las imágenes válidas

# Separar los datos en entrenamiento y validación
train_data, val_data = train_test_split(valid_data, test_size=0.2, random_state=42)

# Crear un generador de imágenes para entrenamiento y validación
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_data,
    x_col="Filepath",
    y_col=["Height_cm", "Weight_kg"],  # Etiquetas de altura y peso
    class_mode="raw",  # Para regresión
    target_size=(192, 192),
    batch_size=32
)

val_generator = datagen.flow_from_dataframe(
    dataframe=val_data,
    x_col="Filepath",
    y_col=["Height_cm", "Weight_kg"],
    class_mode="raw",
    target_size=(192, 192),
    batch_size=32
)

# Definir un modelo usando MobileNetV2 preentrenado
base_model = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Congelar las capas preentrenadas

# Añadir capas para la regresión (altura y peso)
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(2)  # La salida tiene dos valores: altura y peso
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Definir callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('modelo_estima_altura_peso.keras', monitor='val_loss', save_best_only=True)

# Entrenar el modelo con callbacks
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,  # Puedes ajustar el número de épocas
    callbacks=[early_stopping, checkpoint]
)

# Guardar el modelo final (en caso de que no se haya guardado ya el mejor)
model.save('modelo_estima_altura_peso_final.keras')

# Preprocesar una imagen nueva para hacer predicciones
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(192, 192))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Agregar dimensión batch
    img_array /= 255.0  # Normalización
    return img_array

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelo_estima_altura_peso.keras')

# Ruta de una imagen nueva para predecir
new_image_path = r"C:\ruta de la imagen"

# Preprocesar la imagen
img_array = preprocess_image(new_image_path)

# Hacer predicción
prediction = model.predict(img_array)

# Mostrar resultados
altura_predicha_cm = prediction[0][0]
peso_predicho_kg = prediction[0][1]
print(f"Altura estimada: {altura_predicha_cm:.2f} cm")
print(f"Peso estimado: {peso_predicho_kg:.2f} kg")