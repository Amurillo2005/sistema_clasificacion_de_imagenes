"""
Modelo v2: transferencia de aprendizaje por extracción de características.

ResNet50V2 preentrenada con ImageNet se utiliza congelada como extractor
de características. Sobre ella se entrena una cabeza clasificadora para
las 10 categorías de CIFAR-10.

Distribución de datos:
- 40.000 imágenes para entrenamiento.
- 10.000 imágenes para validación.
- 10.000 imágenes oficiales para prueba, reservadas para evaluar
  el modelo una vez finalizado el entrenamiento.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

SEMILLA = 42
tf.random.set_seed(SEMILLA)
np.random.seed(SEMILLA)

TAM = 160

# ---------------------------------------------------------------
# 1. Cargar y dividir los datos
# ---------------------------------------------------------------

datos = np.load("cifar10.npz")

Xtr, ytr = datos["Xtr"], datos["ytr"]
Xte, yte = datos["Xte"], datos["yte"]

# Dividir únicamente el conjunto oficial de entrenamiento.
# La división es estratificada para conservar la proporción de clases.
X_entreno, X_val, y_entreno_original, y_val = train_test_split(
    Xtr,
    ytr,
    test_size=0.20,
    random_state=SEMILLA,
    stratify=ytr.reshape(-1)
)

print("Entrenamiento:", X_entreno.shape)
print("Validación:", X_val.shape)
print("Prueba:", Xte.shape)

# ---------------------------------------------------------------
# 2. Base ResNet50V2 preentrenada y congelada
# ---------------------------------------------------------------

base = tf.keras.applications.ResNet50V2(
    weights=None,
    include_top=False,
    input_shape=(TAM, TAM, 3)
)

base.load_weights("resnet50v2_notop.h5")
base.trainable = False

extractor = models.Sequential([
    layers.Input(shape=(TAM, TAM, 3)),
    layers.Rescaling(1.0 / 127.5, offset=-1.0),
    base,
    layers.GlobalAveragePooling2D(),
], name="extractor")


def extraer(X, espejo=False):
    ds = tf.data.Dataset.from_tensor_slices(X).batch(256)

    def prep(x):
        x = tf.image.resize(
            tf.cast(x, tf.float32),
            (TAM, TAM)
        )

        if espejo:
            x = tf.image.flip_left_right(x)

        return x

    ds = ds.map(
        prep,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(2)

    return extractor.predict(ds, verbose=2)


# ---------------------------------------------------------------
# 3. Extraer características
# ---------------------------------------------------------------

print("\nExtrayendo características...")

F_entreno_original = extraer(X_entreno)
F_entreno_espejado = extraer(X_entreno, espejo=True)
F_val = extraer(X_val)

# Las versiones espejadas solo se generan para entrenamiento.
F_entreno = np.concatenate([
    F_entreno_original,
    F_entreno_espejado
])

y_entreno = np.concatenate([
    y_entreno_original,
    y_entreno_original
])

print("\nCaracterísticas de entrenamiento:", F_entreno.shape)
print("Características de validación:", F_val.shape)

# ---------------------------------------------------------------
# 4. Crear y entrenar la cabeza clasificadora
# ---------------------------------------------------------------

cabeza = models.Sequential([
    layers.Input(shape=(F_entreno.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
], name="cabeza")

cabeza.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cb = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=3,
        min_lr=1e-5
    ),
    tf.keras.callbacks.CSVLogger("historial_v2.csv")
]

historial = cabeza.fit(
    F_entreno,
    y_entreno,
    validation_data=(F_val, y_val),
    epochs=60,
    batch_size=256,
    callbacks=cb,
    verbose=2
)

# ---------------------------------------------------------------
# 5. Guardar la cabeza clasificadora
# ---------------------------------------------------------------

cabeza.save("cabeza_v2.keras")

# ---------------------------------------------------------------
# 6. Ensamblar y guardar el modelo completo
# ---------------------------------------------------------------

entrada = layers.Input(shape=(TAM, TAM, 3))

x = layers.Rescaling(
    1.0 / 127.5,
    offset=-1.0
)(entrada)

x = base(x)
x = layers.GlobalAveragePooling2D()(x)

salida = cabeza(x)

modelo = models.Model(
    entrada,
    salida,
    name="CNN_TransferLearning_v2"
)

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

modelo.save("modelo_cnn_v2.keras")


# ---------------------------------------------------------------
# 7. Conteo de parámetros del modelo completo
# ---------------------------------------------------------------

parametros_entrenables = sum(
    np.prod(variable.shape)
    for variable in modelo.trainable_weights
)

parametros_no_entrenables = sum(
    np.prod(variable.shape)
    for variable in modelo.non_trainable_weights
)

parametros_totales = modelo.count_params()

print("\n===== PARÁMETROS DEL MODELO V2 =====")
print(f"Parámetros entrenables: {parametros_entrenables:,}")
print(f"Parámetros no entrenables: {parametros_no_entrenables:,}")
print(f"Parámetros totales: {parametros_totales:,}")

# Guardar los parámetros en formato JSON
parametros = {
    "parametros_entrenables": int(parametros_entrenables),
    "parametros_no_entrenables": int(parametros_no_entrenables),
    "parametros_totales": int(parametros_totales)
}

with open("parametros_modelo_v2.json", "w", encoding="utf-8") as archivo:
    import json
    json.dump(
        parametros,
        archivo,
        indent=4,
        ensure_ascii=False
    )

print("\nParámetros guardados en parametros_modelo_v2.json")
print("\nEntrenamiento finalizado.")
print("Se guardaron cabeza_v2.keras y modelo_cnn_v2.keras.")
print("El conjunto oficial de prueba no se utilizó durante el entrenamiento.")