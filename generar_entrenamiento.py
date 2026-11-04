
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Leer el historial del entrenamiento
historial = pd.read_csv("historial_v2.csv")

# Mostrar las columnas disponibles
print("Columnas del historial:", historial.columns.tolist())

# Gráfica de exactitud por época
plt.figure(figsize=(9, 5))

plt.plot(
    historial["epoch"] + 1,
    historial["accuracy"],
    label="Entrenamiento"
)

plt.plot(
    historial["epoch"] + 1,
    historial["val_accuracy"],
    label="Validación"
)

plt.title("Exactitud por época - ResNet50V2")
plt.xlabel("Época")
plt.ylabel("Exactitud")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("curva_exactitud_v2.png", dpi=150)
plt.close()


# Gráfica de pérdida por época
plt.figure(figsize=(9, 5))

plt.plot(
    historial["epoch"] + 1,
    historial["loss"],
    label="Entrenamiento"
)

plt.plot(
    historial["epoch"] + 1,
    historial["val_loss"],
    label="Validación"
)

plt.title("Pérdida por época - ResNet50V2")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("curva_perdida_v2.png", dpi=150)
plt.close()

print("\nGráficas generadas correctamente:")
print("- curva_exactitud_v2.png")
print("- curva_perdida_v2.png")
