# API Body Scanner

Esta API permite estimar la altura y el peso de una persona a partir de una imagen enviada. Utiliza un modelo entrenado con `TensorFlow` y procesa las imágenes a través de Flask.

## Requisitos

1. **Python 3.8 o superior**
2. **TensorFlow** (Versión compatible con tu modelo)
3. **Flask** (Para ejecutar la API)
4. **Pillow** (Para manejar la manipulación de imágenes)
5. **Werkzeug** (Para manejo de archivos subidos)

## Instalación

### Paso 1: Clonar el repositorio

Clona el repositorio desde GitHub:

  bash
  git clone https://github.com/Nebuluxe/API-Body-Sacenner.git
  cd API-Body-Sacenner


Paso 2: Crear un entorno virtual

Se recomienda crear un entorno virtual para manejar las dependencias. Entra en el directorio del proyecto y crea un entorno virtual.

bash

python -m venv venv

Activa el entorno virtual:

    En Windows:

    bash

venv\Scripts\activate

En Linux o macOS:

bash

    source venv/bin/activate

Paso 3: Instalar dependencias

Instala todas las dependencias del proyecto ejecutando:

bash

    pip install -r requirements.txt

Asegúrate de que el archivo requirements.txt incluya las siguientes dependencias:

    Flask
    tensorflow
    Pillow
    Werkzeug

Paso 4: Verifica el modelo entrenado

El archivo modelo_estima_altura_peso.keras debe estar en el directorio raíz del proyecto. Si aún no tienes este modelo, asegúrate de entrenarlo o conseguir una copia del modelo.
Paso 5: Ejecutar la API

Una vez que hayas configurado todo, puedes ejecutar la API:

    bash
    
    flask --app api_body_scanner run

Esto ejecutará la API en http://127.0.0.1:5000/.
Uso de la API
Endpoint /predict

Este endpoint espera una imagen enviada mediante una solicitud POST para realizar una predicción de la altura y el peso.
Ejemplo de solicitud con Postman o cURL:

    URL: http://127.0.0.1:5000/predict
    Método: POST
    Parámetro:
        image (archivo de imagen en formatos jpg, jpeg, o png)

    bash
    
    curl -X POST http://127.0.0.1:5000/predict \
      -H 'Content-Type: multipart/form-data' \
      -F 'image=@/ruta/a/tu/imagen.jpg'

Respuesta exitosa:
    
    json
    
    {
      "altura_cm": 170.45,
      "peso_kg": 65.12
    }

Errores comunes:

    Formato de imagen no permitido:

    json

    {
      "error": "Invalid image format. Allowed formats are png, jpg, jpeg"
    }

    Imagen no válida:
    
    json

    {
      "error": "Invalid image file"
    }

Validación de imágenes

La API valida que las imágenes tengan una resolución mínima de 800x1200 y un formato correcto (JPEG o PNG). Además, puedes agregar validaciones adicionales como comprobar si la imagen contiene a una persona o verificar la proporción de la imagen.
Mejoras futuras

    Validación de contenido de imagen: Integrar un modelo de detección de objetos para asegurarse de que la imagen contenga una persona antes de procesarla.
    Interfaz gráfica: Crear una interfaz web para facilitar el uso de la API.

Para que el modelo capture la altura de manera más precisa a partir de una foto, hay varios factores clave que deben considerarse en cuanto a la calidad y características de la imagen. Aquí te dejo una lista de requisitos que la foto debería cumplir para mejorar la precisión del modelo al estimar la altura:

1. Posición del cuerpo:

    Vista frontal o lateral: La persona debe estar de pie en una vista frontal o lateral completa. Las posiciones inclinadas o semi-sentadas pueden afectar negativamente la estimación.
    Postura recta: La persona debe estar completamente erguida y no encorvada para obtener una representación precisa de la altura.

2. Distancia a la cámara:

    Distancia fija: Idealmente, la cámara debe estar a una distancia conocida, preferentemente entre 1.5 y 2 metros. Las distancias demasiado cortas o largas pueden afectar las proporciones y hacer que el modelo estime incorrectamente la altura.
    No inclinación de la cámara: La cámara debe estar a la altura del torso de la persona y no inclinada hacia arriba o hacia abajo. Cualquier ángulo inclinado distorsiona las proporciones.

3. Fondo y condiciones de luz:

    Fondo claro y sin distracciones: Lo ideal es que la persona esté frente a un fondo simple y claro, como una pared blanca o lisa, para que el modelo se centre en el contorno del cuerpo. Los fondos con muchas distracciones pueden confundir al algoritmo.
    Buena iluminación: La iluminación debe ser uniforme y adecuada para evitar sombras profundas que puedan distorsionar la percepción del cuerpo. Una luz suave y difusa es lo mejor para evitar sombras fuertes.

4. Visibilidad del cuerpo completo:

    Cuerpo completo en la imagen: La imagen debe incluir el cuerpo completo, desde la cabeza hasta los pies, sin recortes. Los pies deben ser claramente visibles.
    Sin obstrucciones: La persona no debe estar obstruida por objetos o partes de otras personas. Toda la silueta debe ser visible para que el modelo pueda detectar correctamente la altura.

5. Resolución y calidad de la imagen:

    Alta resolución: La imagen debe tener suficiente resolución para que el modelo pueda detectar detalles precisos. Imágenes borrosas o pixeladas pueden llevar a errores en las predicciones.
    Formato aceptado: Utiliza formatos de imagen estándar como JPG o PNG, y evita imágenes muy comprimidas o en formatos no comunes.

6. Ropa y accesorios:

    Ropa ajustada: Se recomienda que la persona use ropa ajustada para que el contorno del cuerpo sea visible. Ropa muy suelta o abrigos grandes pueden afectar la detección de los límites del cuerpo.
    Sin accesorios grandes: Se deben evitar accesorios como mochilas, sombreros grandes o elementos que cambien significativamente la forma o contorno del cuerpo.

7. Posición de los pies:

    Pies alineados con el suelo: Asegúrate de que los pies estén completamente alineados y apoyados en el suelo, visibles en la foto. Cualquier ángulo o elevación de los pies puede alterar la estimación.

8. No distorsión de lente:

    Evitar lentes gran angular: No utilices lentes que distorsionen la imagen (como lentes gran angulares o "fish-eye"). Las lentes que causan distorsión pueden afectar el cálculo de las proporciones del cuerpo.

Resumen rápido:

    Vista frontal/lateral completa, con el cuerpo erguido.
    Cámara a 1.5-2 metros de distancia.
    Buena iluminación y fondo claro.
    Sin recortes ni obstrucciones.
    Alta resolución, sin distorsiones.
    Ropa ajustada y pies visibles.
