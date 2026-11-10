# Sistema de clasificación de imágenes con CNN y aplicación web

Implementación del proyecto de grado **"Desarrollo de un sistema de clasificación
de imágenes mediante una red neuronal convolucional e integración en una
aplicación web"** — Andrés Caleb Murillo Cervantes y Jesús Alberto Reales Ortiz,
Universidad Libre, seccional Barranquilla.

El sistema clasifica imágenes en **10 categorías** (avión, automóvil, pájaro,
gato, ciervo, perro, rana, caballo, barco y camión) usando una red neuronal
convolucional (CNN) entrenada con el dataset **CIFAR-10** (60.000 imágenes),
integrada a una aplicación web local donde el usuario sube una imagen y recibe
la predicción.




## Requisitos

- Python 3.13.5
- Conexión a internet solo para instalar las librerías la primera vez

## Cómo ejecutar la aplicación web (paso a paso)

1. Abre una terminal dentro de esta carpeta.

2. Instala las dependencias (solo la primera vez):

   ```bash
   pip install -r requirements.txt
   ```

   > En Windows usa `pip` o `py -m pip`. Si `tensorflow-cpu` falla,
   > prueba con `pip install tensorflow`.

3. Arranca el servidor:

   ```bash
   python app.py
   ```

4. Abre tu navegador en **http://localhost:5000**

5. Sube cualquier imagen (una foto de un perro, un carro, un avión…),
   pulsa **«Clasificar imagen»** y verás la categoría predicha con su nivel
   de confianza y las 5 probabilidades más altas.

Para detener el servidor: `Ctrl + C` en la terminal.

## Contenido de la carpeta

| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación web Flask que carga el modelo y responde las predicciones |
| `templates/index.html` | Interfaz web (subida de imagen, vista previa y resultados) |
| `cabeza_v2.keras` | Cabeza clasificadora del modelo v2 ya **entrenada** — la que usa la app |
| `modelo_v2.py` | Arma el modelo v2 (descarga la base ResNet50V2 la primera vez) |
| `clases.json` | Nombres de las 10 categorías |
| `entrenar_v2.py` | Scripts con los que se entrenó el modelo V2 |
| `evaluar_v2.py` | Script de evaluación para el modelo V2: métricas y matriz de confusión |
| `metricas.json` | Métrica del modelo en el conjunto de prueba |
| `historial_v2.csv` | Exactitud y pérdida de cada época |
| `generar_entrenamiento.py` | Genera las gráficas de exactitud y pérdida por época |
| `curva_exactitud_v2.png` | Gráfica de la exactitud del modelo por época |
| `curva_perdida_v2.png` | Gráfica de la pérdida del modelo por época |
| `matriz_confusion_v2.png` | Matriz de confusión del modelo |

## Arquitectura

**Modelo — transferencia de aprendizaje**: se usa **ResNet50V2**, una CNN
preentrenada con ImageNet (1.4 millones de fotos reales), congelada como
extractor de características; sobre ella se entrena una cabeza clasificadora
propia (Densa 256 + softmax 10) con las imágenes de CIFAR-10 llevadas a
160×160. Por eso reconoce mucho mejor las fotografías reales que sube el
usuario.

```
Entrada 160×160×3
→ ResNet50V2 preentrenada (congelada)   (características visuales generales)
→ GlobalAveragePooling → vector 2048
→ Densa 256 → Softmax 10                (clasificación propia)
```

## Re-entrenar los modelos (opcional)

El entrenamiento usa el dataset CIFAR-10 en formato `cifar10.npz`
(no incluido en el zip por su tamaño). Para regenerarlo:

```bash
git clone --depth 1 https://github.com/YoongiKim/CIFAR-10-images.git cifar10-images
python preparar_datos.py   # crea cifar10.npz
python entrenar_v2.py      # entrena y guarda modelo_cnn_v2.keras (v2)
python evaluar_v2.py       # métricas y gráficas del v2
```

Para el modelo hacen falta además los pesos preentrenados de ResNet50V2
(`resnet50v2_notop.h5`), que se descargan de:
`https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5`

## Notas

- La imagen subida se convierte a RGB, se redimensiona al tamaño de entrada
  del modelo y se normaliza antes de entrar a la red, igual que en el
  entrenamiento.
- El modelo solo conoce las 10 categorías de CIFAR-10: si subes algo distinto
  (una pizza, una persona), responderá con la categoría que más se le parezca.
- Funciona mejor con fotos donde el objeto ocupa la mayor parte de la imagen.
- La primera predicción tras arrancar el servidor tarda unos segundos
  (el modelo se "calienta"); las siguientes son casi instantáneas.
