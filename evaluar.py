
# -*- coding: utf-8 -*-
"""
Evaluación del modelo CNN V1:
exactitud, precisión, recall, F1-macro y matriz de confusión.
"""

import json
import csv

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ---------------------------------------------------------------
# 1. Carga de clases, datos y modelo
# ---------------------------------------------------------------

with open("clases.json", encoding="utf-8") as f:
    CLASES = json.load(f)

datos = np.load("cifar10.npz")

# Exclusivamente el conjunto oficial de prueba
Xte = datos["Xte"].astype("float32") / 255.0
yte = datos["yte"].reshape(-1)

modelo = tf.keras.models.load_model("modelo_cnn.keras")

# ---------------------------------------------------------------
# 2. Conteo de parámetros del modelo guardado
# ---------------------------------------------------------------

parametros_entrenables = sum(
    np.prod(variable.shape)
    for variable in modelo.trainable_weights
)

parametros_no_entrenables = sum(
    np.prod(variable.shape)
    for variable in modelo.non_trainable_weights
)

print("===== PARÁMETROS DEL MODELO V1 =====")
print(f"Entrenables: {parametros_entrenables:,}")
print(f"No entrenables: {parametros_no_entrenables:,}")
print(f"Totales: {modelo.count_params():,}")

# ---------------------------------------------------------------
# 3. Predicciones y métricas
# ---------------------------------------------------------------

pred = np.argmax(
    modelo.predict(Xte, batch_size=256, verbose=0),
    axis=1
)

exactitud = accuracy_score(yte, pred)

precision = precision_score(
    yte, pred, average="macro"
)

recall = recall_score(
    yte, pred, average="macro"
)

f1 = f1_score(
    yte, pred, average="macro"
)

print("\n===== MÉTRICAS SOBRE EL CONJUNTO OFICIAL DE PRUEBA =====")
print(f"Número de imágenes: {len(Xte)}")
print(f"Exactitud (accuracy): {exactitud:.4f}")
print(f"Precisión (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1-score (macro): {f1:.4f}")

print("\n===== INFORME DE CLASIFICACIÓN =====")
print(
    classification_report(
        yte,
        pred,
        target_names=CLASES,
        digits=4
    )
)

# ---------------------------------------------------------------
# 4. Guardar métricas
# ---------------------------------------------------------------

with open("metricas.json", "w", encoding="utf-8") as f:
    json.dump({
        "exactitud": round(float(exactitud), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "parametros_entrenables": int(parametros_entrenables),
        "parametros_no_entrenables": int(parametros_no_entrenables),
        "parametros_totales": int(modelo.count_params())
    }, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------
# 5. Matriz de confusión
# ---------------------------------------------------------------

mc = confusion_matrix(yte, pred)

np.savetxt(
    "matriz_confusion.csv",
    mc,
    fmt="%d",
    delimiter=","
)

# ---------------------------------------------------------------
# 6. Gráficas de entrenamiento y validación
# ---------------------------------------------------------------

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

epocas = []
acc = []
val_acc = []
perdida = []
val_perdida = []

with open("historial_entrenamiento.csv", newline="") as f:
    for fila in csv.DictReader(f):
        epocas.append(int(fila["epoch"]) + 1)
        acc.append(float(fila["accuracy"]))
        val_acc.append(float(fila["val_accuracy"]))
        perdida.append(float(fila["loss"]))
        val_perdida.append(float(fila["val_loss"]))

AZUL = "#4269d0"
GRIS = "#3ca951"

fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

ejes[0].plot(epocas, acc, color=AZUL, label="Entrenamiento")
ejes[0].plot(epocas, val_acc, color=GRIS, label="Validación")
ejes[0].set_title("Exactitud por época")
ejes[0].set_xlabel("Época")
ejes[0].set_ylabel("Exactitud")

ejes[1].plot(epocas, perdida, color=AZUL, label="Entrenamiento")
ejes[1].plot(epocas, val_perdida, color=GRIS, label="Validación")
ejes[1].set_title("Pérdida por época")
ejes[1].set_xlabel("Época")
ejes[1].set_ylabel("Pérdida")

for eje in ejes:
    eje.legend(frameon=False)
    eje.spines[["top", "right"]].set_visible(False)
    eje.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig("curvas_entrenamiento.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# 7. Gráfica de matriz de confusión
# ---------------------------------------------------------------

fig2, eje = plt.subplots(figsize=(7.2, 6), dpi=150)

im = eje.imshow(mc, cmap="Blues")

eje.set_xticks(range(10), CLASES, rotation=45, ha="right")
eje.set_yticks(range(10), CLASES)

eje.set_xlabel("Clase predicha")
eje.set_ylabel("Clase real")
eje.set_title("Matriz de confusión — conjunto de prueba")

umbral = mc.max() / 2

for i in range(10):
    for j in range(10):
        eje.text(
            j,
            i,
            mc[i, j],
            ha="center",
            va="center",
            fontsize=7,
            color="white" if mc[i, j] > umbral else "#334155"
        )

fig2.colorbar(im, shrink=0.8)
fig2.tight_layout()
fig2.savefig("matriz_confusion.png", bbox_inches="tight")
plt.close(fig2)

print("\nArchivos generados:")
print("- metricas.json")
print("- parametros_modelo.json")
print("- matriz_confusion.csv")
print("- curvas_entrenamiento.png")
print("- matriz_confusion.png")
