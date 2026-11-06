
import json
import numpy as np
import tensorflow as tf

# Cargar el modelo completo
modelo = tf.keras.models.load_model("modelo_cnn_v2.keras")

# Contar parámetros
parametros_entrenables = sum(
    np.prod(variable.shape)
    for variable in modelo.trainable_weights
)

parametros_no_entrenables = sum(
    np.prod(variable.shape)
    for variable in modelo.non_trainable_weights
)

parametros_totales = modelo.count_params()

# Mostrar resultados
print("===== PARÁMETROS DEL MODELO RESNET50V2 =====")
print(f"Entrenables: {parametros_entrenables:,}")
print(f"No entrenables: {parametros_no_entrenables:,}")
print(f"Totales: {parametros_totales:,}")

# Guardar resultados
parametros = {
    "parametros_entrenables": int(parametros_entrenables),
    "parametros_no_entrenables": int(parametros_no_entrenables),
    "parametros_totales": int(parametros_totales)
}

with open("parametros_modelo_v2.json", "w", encoding="utf-8") as archivo:
    json.dump(parametros, archivo, indent=4, ensure_ascii=False)

print("\nArchivo parametros_modelo_v2.json generado.")
