
# -*- coding: utf-8 -*-

"""
Entrenamiento del modelo CNN V1 para clasificación de imágenes CIFAR-10.

Proyecto de grado:
Desarrollo de un sistema de clasificación de imágenes mediante
una red neuronal convolucional y despliegue en una aplicación web.

Protocolo experimental:
- 40.000 imágenes para entrenamiento.
- 10.000 imágenes para validación.
- 10.000 imágenes para prueba.
- Aumento de datos mediante volteo horizontal fijo.
- 80.000 muestras finales para entrenamiento.
"""

import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)

from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

SEMILLA = 42

np.random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)

NOMBRE_MODELO = "modelo_cnn.keras"
NOMBRE_HISTORIAL = "historial_entrenamiento.csv"

EPOCAS = 25
TAMANO_LOTE = 128


# ============================================================
# 2. CARGA DEL CONJUNTO DE DATOS
# ============================================================

print("\nCargando conjunto de datos CIFAR-10...")

datos = np.load("cifar10.npz")

Xtr = datos["Xtr"]
ytr = datos["ytr"]

Xte = datos["Xte"]
yte = datos["yte"]

# Normalización de las imágenes
Xtr = Xtr.astype("float32") / 255.0
Xte = Xte.astype("float32") / 255.0

print("Imágenes oficiales de entrenamiento:", Xtr.shape)
print("Etiquetas de entrenamiento:", ytr.shape)
print("Imágenes oficiales de prueba:", Xte.shape)
print("Etiquetas de prueba:", yte.shape)


# ============================================================
# 3. PARTICIÓN DE ENTRENAMIENTO Y VALIDACIÓN
# ============================================================

print("\nSeparando los datos para entrenamiento y validación...")

X_entreno, X_val, y_entreno, y_val = train_test_split(
    Xtr,
    ytr,
    test_size=0.20,
    random_state=SEMILLA,
    stratify=ytr.reshape(-1)
)

print("Imágenes para entrenamiento:", X_entreno.shape)
print("Imágenes para validación:", X_val.shape)
print("Imágenes para prueba:", Xte.shape)


# ============================================================
# 4. AUMENTO DE DATOS: VOLTEO HORIZONTAL FIJO
# ============================================================

print("\nAplicando aumento de datos mediante volteo horizontal...")

# Generar versiones volteadas horizontalmente.
# El eje 2 corresponde al ancho de las imágenes CIFAR-10.
X_entreno_espejo = np.flip(
    X_entreno,
    axis=2
).copy()

# Combinar imágenes originales y sus versiones volteadas.
X_entreno_aumentado = np.concatenate(
    [X_entreno, X_entreno_espejo],
    axis=0
)

# Duplicar las etiquetas correspondientes.
y_entreno_aumentado = np.concatenate(
    [y_entreno, y_entreno],
    axis=0
)

print("Imágenes originales:", X_entreno.shape[0])
print("Imágenes volteadas:", X_entreno_espejo.shape[0])
print("Total de muestras aumentadas:", X_entreno_aumentado.shape[0])


# ============================================================
# 5. CONSTRUCCIÓN DE LA ARQUITECTURA CNN V1
# ============================================================

print("\nConstruyendo el modelo CNN V1...")

entrada = layers.Input(shape=(32, 32, 3))

x = entrada

# ------------------------------------------------------------
# Bloque convolucional 1
# ------------------------------------------------------------

x = layers.Conv2D(
    32,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Conv2D(
    32,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D((2, 2))(x)

x = layers.Dropout(0.25)(x)


# ------------------------------------------------------------
# Bloque convolucional 2
# ------------------------------------------------------------

x = layers.Conv2D(
    64,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Conv2D(
    64,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D((2, 2))(x)

x = layers.Dropout(0.25)(x)


# ------------------------------------------------------------
# Bloque convolucional 3
# ------------------------------------------------------------

x = layers.Conv2D(
    128,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Conv2D(
    128,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D((2, 2))(x)

x = layers.Dropout(0.3)(x)


# ------------------------------------------------------------
# Clasificador
# ------------------------------------------------------------

x = layers.Flatten()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.4)(x)

salida = layers.Dense(
    10,
    activation="softmax"
)(x)


# Construcción del modelo
modelo = models.Model(
    inputs=entrada,
    outputs=salida
)


# ============================================================
# 6. COMPILACIÓN DEL MODELO
# ============================================================

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 7. CONTEO DE PARÁMETROS
# ============================================================

parametros_entrenables = sum(
    np.prod(variable.shape)
    for variable in modelo.trainable_weights
)

parametros_no_entrenables = sum(
    np.prod(variable.shape)
    for variable in modelo.non_trainable_weights
)

parametros_totales = modelo.count_params()

print("\nResumen de la arquitectura:")
modelo.summary()

print("\nCantidad de parámetros:")
print("Parámetros entrenables:", parametros_entrenables)
print("Parámetros no entrenables:", parametros_no_entrenables)
print("Parámetros totales:", parametros_totales)


# Guardar los parámetros en un archivo JSON.
parametros = {
    "parametros_entrenables": int(parametros_entrenables),
    "parametros_no_entrenables": int(parametros_no_entrenables),
    "parametros_totales": int(parametros_totales)
}

with open("parametros_modelo.json", "w", encoding="utf-8") as archivo:
    json.dump(
        parametros,
        archivo,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 8. CONFIGURACIÓN DE LOS CALLBACKS
# ============================================================

callbacks = [

    # Guardar el modelo con mejor exactitud de validación.
    ModelCheckpoint(
        filepath=NOMBRE_MODELO,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    # Reducir la tasa de aprendizaje cuando la validación
    # no presente mejoras durante varias épocas.
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),

    # Registrar el historial del entrenamiento.
    CSVLogger(
        NOMBRE_HISTORIAL,
        append=False
    )
]


# ============================================================
# 9. ENTRENAMIENTO DEL MODELO
# ============================================================

print("\nIniciando entrenamiento del modelo CNN V1...")

historial = modelo.fit(
    X_entreno_aumentado,
    y_entreno_aumentado,
    validation_data=(X_val, y_val),
    epochs=EPOCAS,
    batch_size=TAMANO_LOTE,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# 10. EVALUACIÓN FINAL CON EL CONJUNTO DE PRUEBA
# ============================================================

print("\nCargando el mejor modelo obtenido durante el entrenamiento...")

mejor_modelo = tf.keras.models.load_model(
    NOMBRE_MODELO
)

print("\nEvaluando el modelo con el conjunto oficial de prueba...")

resultados = mejor_modelo.evaluate(
    Xte,
    yte,
    verbose=1
)

print("\nResultados de evaluación:")
print("Pérdida en prueba:", resultados[0])
print("Exactitud en prueba:", resultados[1])


# ============================================================
# 11. GUARDAR LAS CLASES
# ============================================================

clases = [
    "avion",
    "automovil",
    "ave",
    "gato",
    "ciervo",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camion"
]

with open("clases.json", "w", encoding="utf-8") as archivo:
    json.dump(
        clases,
        archivo,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 12. FINALIZACIÓN
# ============================================================

print("\nEntrenamiento finalizado correctamente.")

print("Modelo guardado:", NOMBRE_MODELO)
print("Historial guardado:", NOMBRE_HISTORIAL)
print("Parámetros guardados: parametros_modelo.json")
print("Clases guardadas: clases.json")
