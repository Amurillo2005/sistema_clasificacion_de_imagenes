
import numpy as np
import tensorflow as tf

print("Cargando el conjunto de datos CIFAR-10...")

# Descargar o cargar CIFAR-10 desde la caché de Keras
(Xtr, ytr), (Xte, yte) = tf.keras.datasets.cifar10.load_data()

# Guardar los datos con los nombres que espera entrenar_v2.py
np.savez_compressed(
    "cifar10.npz",
    Xtr=Xtr,
    ytr=ytr,
    Xte=Xte,
    yte=yte
)

print("\nDatos preparados correctamente.")
print("Entrenamiento:", Xtr.shape, ytr.shape)
print("Prueba:", Xte.shape, yte.shape)
print("Archivo generado: cifar10.npz")
