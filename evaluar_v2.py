
"""
Evaluación final del modelo V2 mediante el conjunto oficial de prueba.
Genera métricas globales, métricas por categoría y matriz de confusión.
"""

import csv
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from matplotlib.ticker import PercentFormatter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# 1. Cargar clases y conjunto oficial de prueba
# ---------------------------------------------------------------

with open("clases.json", encoding="utf-8") as archivo:
    CLASES = json.load(archivo)

datos = np.load("cifar10.npz")

Xte = datos["Xte"]
yte = datos["yte"].reshape(-1)

print("Imágenes de prueba:", Xte.shape)
print("Etiquetas de prueba:", yte.shape)


# ---------------------------------------------------------------
# 2. Cargar modelo y realizar predicciones
# ---------------------------------------------------------------

modelo = tf.keras.models.load_model("modelo_cnn_v2.keras")
TAM = modelo.input_shape[1]

ds = (
    tf.data.Dataset.from_tensor_slices(Xte)
    .batch(128)
    .map(
        lambda x: tf.image.resize(
            tf.cast(x, tf.float32),
            (TAM, TAM)
        )
    )
)

probabilidades = modelo.predict(ds, verbose=1)
pred = np.argmax(probabilidades, axis=1)


# ---------------------------------------------------------------
# 3. Calcular métricas globales
# ---------------------------------------------------------------

exactitud = accuracy_score(yte, pred)

precision = precision_score(
    yte, pred, average="macro", zero_division=0
)

recall = recall_score(
    yte, pred, average="macro", zero_division=0
)

f1 = f1_score(
    yte, pred, average="macro", zero_division=0
)

metricas = {
    "cantidad_imagenes_prueba": int(len(yte)),
    "exactitud": round(float(exactitud), 4),
    "precision_macro": round(float(precision), 4),
    "recall_macro": round(float(recall), 4),
    "f1_macro": round(float(f1), 4)
}

print("\n===== EVALUACIÓN FINAL DEL MODELO V2 =====")
print(f"Imágenes evaluadas:          {len(yte)}")
print(f"Exactitud (accuracy):        {exactitud:.4f}")
print(f"Precisión (macro):           {precision:.4f}")
print(f"Recall (macro):              {recall:.4f}")
print(f"F1-score (macro):            {f1:.4f}")


# Guardar métricas globales
with open("metricas.json", "w", encoding="utf-8") as archivo:
    json.dump(
        metricas,
        archivo,
        ensure_ascii=False,
        indent=4
    )


# ---------------------------------------------------------------
# 4. Generar informe de clasificación por categoría
# ---------------------------------------------------------------

reporte = classification_report(
    yte,
    pred,
    target_names=CLASES,
    digits=4,
    zero_division=0
)

print("\n===== MÉTRICAS POR CATEGORÍA =====")
print(reporte)

with open(
    "reporte_clasificacion.txt",
    "w",
    encoding="utf-8"
) as archivo:
    archivo.write(reporte)


# ---------------------------------------------------------------
# 5. Generar matriz de confusión
# ---------------------------------------------------------------

mc = confusion_matrix(yte, pred)

# Matriz de confusión normalizada (porcentajes)
mc_normalizada = confusion_matrix(
    yte,
    pred,
    normalize="true"
)

# Guardar matriz en CSV con nombres de categorías
with open(
    "matriz_confusion_v2.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as archivo:
    escritor = csv.writer(archivo)

    escritor.writerow(["Clase real / Clase predicha"] + CLASES)

    for nombre, fila in zip(CLASES, mc):
        escritor.writerow([nombre] + fila.tolist())

# Guardar matriz normalizada en CSV
with open(
    "matriz_confusion_normalizada_v2.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as archivo:
    escritor = csv.writer(archivo)

    escritor.writerow(["Clase real / Clase predicha"] + CLASES)

    for nombre, fila in zip(CLASES, mc_normalizada):
        escritor.writerow([nombre] + fila.tolist())

# ---------------------------------------------------------------
# 6. Visualizar y guardar matrices de confusión
# ---------------------------------------------------------------

# 6.1 Matriz de confusión original

fig, eje = plt.subplots(figsize=(8, 7), dpi=150)

im = eje.imshow(mc, cmap="Blues")

eje.set_xticks(range(len(CLASES)))
eje.set_xticklabels(CLASES, rotation=45, ha="right")

eje.set_yticks(range(len(CLASES)))
eje.set_yticklabels(CLASES)

eje.set_xlabel("Clase predicha")
eje.set_ylabel("Clase real")
eje.set_title("Matriz de confusión - ResNet50V2")

umbral = mc.max() / 2

for i in range(len(CLASES)):
    for j in range(len(CLASES)):
        eje.text(
            j,
            i,
            mc[i, j],
            ha="center",
            va="center",
            fontsize=7,
            color="white" if mc[i, j] > umbral else "#334155"
        )

fig.colorbar(im, ax=eje, shrink=0.8)
fig.tight_layout()

fig.savefig(
    "matriz_confusion_v2.png",
    bbox_inches="tight"
)

plt.close(fig)



# 6.2 Matriz de confusión normalizada

fig, eje = plt.subplots(figsize=(8, 7), dpi=150)

im = eje.imshow(
    mc_normalizada,
    cmap="Blues",
    vmin=0,
    vmax=1
)

eje.set_xticks(range(len(CLASES)))
eje.set_xticklabels(CLASES, rotation=45, ha="right")

eje.set_yticks(range(len(CLASES)))
eje.set_yticklabels(CLASES)

eje.set_xlabel("Clase predicha")
eje.set_ylabel("Clase real")
eje.set_title("Matriz de confusión normalizada - ResNet50V2")

umbral = 0.5

for i in range(len(CLASES)):
    for j in range(len(CLASES)):
        valor = mc_normalizada[i, j]

        eje.text(
            j,
            i,
            f"{valor:.1%}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if valor > umbral else "#334155"
        )

fig.colorbar(
    im,
    ax=eje,
    shrink=0.8,
    format=PercentFormatter(xmax=1)
)

fig.tight_layout()

fig.savefig(
    "matriz_confusion_normalizada_v2.png",
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------------
# 7. Confirmar archivos generados
# ---------------------------------------------------------------

print("\nEvaluación finalizada correctamente.")
print("Archivos generados:")
print("- metricas.json")
print("- reporte_clasificacion.txt")
print("- matriz_confusion_v2.csv")
print("- matriz_confusion_v2.png")
print("- matriz_confusion_normalizada_v2.csv")
print("- matriz_confusion_normalizada_v2.png")