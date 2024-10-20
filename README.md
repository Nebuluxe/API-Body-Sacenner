
# API Body Scanner

Esta API permite estimar la altura y el peso de una persona a partir de una imagen enviada. Utiliza un modelo entrenado con `TensorFlow` y procesa las imágenes a través de Flask en ambiente local. Está preparado para que descargues este proyecto y lo subas a tu repositorio de GitHub para hacer un deploy hacia Heroku.

## Requisitos

1. **Python 3.8 o superior**
2. **TensorFlow** (Versión compatible con tu modelo)
3. **Flask** (Para ejecutar la API)
4. **Pillow** (Para manejar la manipulación de imágenes)
5. **Werkzeug** (Para manejo de archivos subidos)
6. **Gunicorn** (Para deploy en producción en Heroku)

## Instalación Local

### Paso 1: Clonar el repositorio

Clona el repositorio desde GitHub:

```bash
git clone https://github.com/Nebuluxe/API-Body-Sacenner.git
cd API-Body-Sacenner
```

### Paso 2: Crear un entorno virtual

Se recomienda crear un entorno virtual para manejar las dependencias. Entra en el directorio del proyecto y crea un entorno virtual.

```bash
python -m venv venv
```

Activa el entorno virtual:

En Windows:
```bash
venv\Scripts\activate
```

En Linux o macOS:
```bash
source venv/bin/activate
```

### Paso 3: Instalar dependencias

Instala todas las dependencias del proyecto ejecutando:

```bash
pip install -r requirements.txt
```

Asegúrate de que el archivo `requirements.txt` incluya las siguientes dependencias:

```
Flask
tensorflow-cpu
Pillow
Werkzeug
gunicorn
```

### Paso 4: Verifica el modelo entrenado

El archivo `modelo_estima_altura_peso.keras` debe estar en el directorio raíz del proyecto. Si aún no tienes este modelo, asegúrate de entrenarlo o conseguir una copia del modelo.

### Paso 5: Ejecutar la API localmente

Una vez que hayas configurado todo, puedes ejecutar la API en modo local con Flask:

```bash
flask --app api_body_scanner run
```

Esto ejecutará la API en http://127.0.0.1:5000/.

## Uso de la API

### Endpoint `/predict`

Este endpoint espera una imagen enviada mediante una solicitud POST para realizar una predicción de la altura y el peso.

Ejemplo de solicitud con Postman o cURL:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@/ruta/a/tu/imagen.jpg'
```

**Respuesta exitosa**:

```json
{
  "altura_cm": 170.45,
  "peso_kg": 65.12
}
```

**Errores comunes**:

Formato de imagen no permitido:
```json
{
  "error": "Invalid image format. Allowed formats are png, jpg, jpeg"
}
```

Imagen no válida:
```json
{
  "error": "Invalid image file"
}
```

## Despliegue en Heroku

### Paso 1: Crear una cuenta en Heroku y configurar Heroku CLI

- Crea una cuenta en [Heroku](https://www.heroku.com).
- Instala [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) en tu máquina.

Inicia sesión en Heroku:

```bash
heroku login
```

### Paso 2: Crear una aplicación en Heroku

Desde la raíz de tu proyecto, ejecuta:

```bash
heroku create nombre-de-tu-app
```

Esto crea una aplicación en Heroku y vincula tu repositorio local con Heroku.

### Paso 3: Configurar el `Procfile`

Asegúrate de que tienes un archivo `Procfile` en la raíz de tu proyecto con el siguiente contenido:

```
web: gunicorn api_body_scanner:app
```

Este archivo le dice a Heroku que use Gunicorn para ejecutar la aplicación Flask.

### Paso 4: Subir el proyecto a GitHub

Si aún no has subido tu proyecto a GitHub, puedes hacerlo ahora:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/tu-usuario/API-Body-Scanner.git
git push -u origin master
```

### Paso 5: Desplegar en Heroku

Con el repositorio local listo, ejecuta los siguientes comandos para desplegar la aplicación en Heroku:

```bash
git add .
git commit -m "Agregar gunicorn para producción"
git push heroku master
```

Heroku instalará automáticamente las dependencias y desplegará tu aplicación.

### Paso 6: Verificar el despliegue

Una vez desplegada la aplicación, puedes abrirla con:

```bash
heroku open
```

También puedes verificar que la API esté en funcionamiento utilizando cURL o Postman apuntando a la URL de Heroku:

```bash
curl -X POST https://nombre-de-tu-app.herokuapp.com/predict \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@/ruta/a/tu/imagen.jpg'
```

### Errores Comunes en Heroku

- **App Crashed**: Verifica los logs de tu aplicación con:
  ```bash
  heroku logs --tail
  ```
- **Slug Size Too Large**: Asegúrate de que el tamaño del proyecto no exceda el límite de 500MB. Si es necesario, usa `.slugignore` para evitar subir archivos grandes innecesarios.

## Validación de Imágenes

La API valida que las imágenes tengan una resolución mínima de 800x1200 y un formato correcto (JPEG o PNG). Además, puedes agregar validaciones adicionales como comprobar si la imagen contiene a una persona o verificar la proporción de la imagen.

## Mejoras Futuras

1. **Validación de contenido de imagen**: Integrar un modelo de detección de objetos para asegurarse de que la imagen contenga una persona antes de procesarla.
2. **Interfaz gráfica**: Crear una interfaz web para facilitar el uso de la API.

---

Con esta guía puedes ejecutar la API tanto en tu entorno local como desplegarla en Heroku fácilmente.
