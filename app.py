# -*- coding: utf-8 -*-
"""
Aplicación web (Flask) que sirve el modelo CNN entrenado para
clasificación de imágenes en tiempo real.

Proyecto de grado: "Desarrollo de un sistema de clasificación de imágenes
mediante una red neuronal convolucional y despliegue en una aplicación web"
Andrés Caleb Murillo Cervantes - Jesús Alberto Reales Ortiz
Universidad Libre, seccional Barranquilla

Uso:
    python app.py
    Luego abrir http://localhost:5000 en el navegador.
"""
import io
import json
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

RUTA = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # máx. 10 MB por imagen

# Carga del modelo entrenado y de las clases (una sola vez, al arrancar).
# Si está disponible el modelo v2 (transferencia de aprendizaje) se usa ese;
# si no, se usa el modelo v1 (CNN propia, imágenes 32x32).
from modelo_v2 import cargar_modelo_v2  # noqa: E402

modelo = cargar_modelo_v2(RUTA)
if modelo is not None:
    TAMANO = modelo.input_shape[1]  # tamaño de entrada del modelo (160)
    ESCALAR = False                 # el v2 normaliza internamente (espera 0-255)
    print(f">> Usando modelo v2 (transferencia de aprendizaje, {TAMANO}x{TAMANO})")
else:
    modelo = tf.keras.models.load_model(os.path.join(RUTA, "modelo_cnn.keras"))
    TAMANO, ESCALAR = 32, True      # el v1 espera valores normalizados 0-1
    print(">> Usando modelo v1 (CNN propia, 32x32)")

with open(os.path.join(RUTA, "clases.json"), encoding="utf-8") as f:
    CLASES = json.load(f)

EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def preprocesar(imagen_bytes):
    """Convierte la imagen subida al formato que espera la CNN:
    RGB, redimensionada al tamaño de entrada del modelo."""
    img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    img = img.resize((TAMANO, TAMANO), Image.BILINEAR)
    arreglo = np.asarray(img, dtype="float32")
    if ESCALAR:
        arreglo = arreglo / 255.0
    return np.expand_dims(arreglo, axis=0)


@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/predecir", methods=["POST"])
def predecir():
    if "imagen" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen."}), 400

    archivo = request.files["imagen"]
    ext = os.path.splitext(archivo.filename or "")[1].lower()
    if ext and ext not in EXTENSIONES:
        return jsonify({"error": f"Formato no soportado: {ext}"}), 400

    try:
        entrada = preprocesar(archivo.read())
    except Exception:
        return jsonify({"error": "El archivo no es una imagen válida."}), 400

    probabilidades = modelo.predict(entrada, verbose=0)[0]
    orden = np.argsort(probabilidades)[::-1]

    resultado = {
        "clase": CLASES[int(orden[0])],
        "confianza": round(float(probabilidades[orden[0]]) * 100, 2),
        "top": [
            {"clase": CLASES[int(i)],
             "probabilidad": round(float(probabilidades[i]) * 100, 2)}
            for i in orden[:5]
        ],
    }
    return jsonify(resultado)


if __name__ == "__main__":
    print("\n  Sistema de clasificación de imágenes con CNN")
    print("  Abre en tu navegador:  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
