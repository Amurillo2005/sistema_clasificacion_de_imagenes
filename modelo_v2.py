# -*- coding: utf-8 -*-
"""
Construcción del modelo v2 (transferencia de aprendizaje).

El modelo v2 = base ResNet50V2 preentrenada (ImageNet) + cabeza propia
entrenada con CIFAR-10. Para que el proyecto no pese ~100 MB, la base
no se incluye: la primera vez que se ejecute la app, este módulo descarga
los pesos (90 MB) directamente desde el repositorio oficial de Keras en
GitHub y los guarda en la carpeta. Las siguientes veces ya no descarga nada.
"""
import os
import urllib.request

import tensorflow as tf
from tensorflow.keras import layers, models

TAM = 160
URL_PESOS = ("https://github.com/keras-team/keras-applications/releases/"
             "download/resnet/"
             "resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5")


def cargar_modelo_v2(carpeta):
    ruta_pesos = os.path.join(carpeta, "resnet50v2_notop.h5")
    ruta_cabeza = os.path.join(carpeta, "cabeza_v2.keras")

    if not os.path.exists(ruta_cabeza):
        return None  # no hay modelo v2 disponible

    if not os.path.exists(ruta_pesos):
        print(">> Primera ejecución: descargando la base preentrenada "
              "ResNet50V2 (90 MB, solo esta vez)...")
        req = urllib.request.Request(URL_PESOS,
                                     headers={"User-Agent": "Mozilla/5.0"})
        datos = urllib.request.urlopen(req, timeout=300).read()
        with open(ruta_pesos, "wb") as f:
            f.write(datos)
        print(">> Descarga completa.")

    base = tf.keras.applications.ResNet50V2(weights=None, include_top=False,
                                            input_shape=(TAM, TAM, 3))
    base.load_weights(ruta_pesos)
    base.trainable = False

    cabeza = tf.keras.models.load_model(ruta_cabeza)

    entrada = layers.Input(shape=(TAM, TAM, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(entrada)  # 0-255 -> [-1,1]
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    salida = cabeza(x)
    return models.Model(entrada, salida, name="CNN_TransferLearning_v2")
