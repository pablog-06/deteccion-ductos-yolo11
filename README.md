# Detección de defectos en tuberías con YOLO11

Versión básica para descargar las imágenes originales, ejecutar un detector ya entrenado y comparar sus predicciones con las etiquetas.

El repositorio incluye **978 imágenes originales con sus etiquetas** y **un único peso entrenado**, `pesos/prueba_02.pt`. Este modelo corresponde al conjunto original de aproximadamente mil imágenes. No incorpora las imágenes externas de Sewer-ML ni las particiones experimentales posteriores.

## Qué se descarga

| Partición | Imágenes | Cajas anotadas | Uso |
| --- | ---: | ---: | --- |
| train | 684 | 1.636 | Entrenamiento |
| valid | 196 | 510 | Supervisión y desarrollo |
| test | 98 | 218 | Evaluación final reservada |
| Total | **978** | **2.364** | Conjunto original |

El peso incluido procede de **prueba_02, de 30 épocas**: entrenó con las 684 imágenes de train y supervisó con las 196 de valid. No entrenó con las 978 imágenes juntas ni con las imágenes nuevas que se investigaron después.

## Descargar código, peso e imágenes

En GitHub puedes elegir **Code → Download ZIP** y descomprimirlo. También puedes usar Git:

```powershell
git clone --depth 1 https://github.com/pablog-06/deteccion-ductos-yolo11.git
cd deteccion-ductos-yolo11
```

Las imágenes y etiquetas se encuentran en [datos/sewer-defects](datos/sewer-defects). El peso está en [pesos/prueba_02.pt](pesos/prueba_02.pt). Son archivos incluidos en la descarga, sin necesidad de obtener el dataset por separado ni usar Git LFS. Si el repositorio es privado, necesitarás acceso concedido por su propietario.

## Instalación en Windows

Se comprobó con Python 3.14.3, PyTorch 2.11.0+cu128, torchvision 0.26.0 y Ultralytics 8.4.190. Los scripts de inferencia y entrenamiento actuales utilizan GPU NVIDIA con CUDA. El dibujo de etiquetas no necesita GPU.

En una instalación nueva, abre PowerShell en la carpeta del proyecto:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

La variante de PyTorch debe corresponder a tu equipo; consulta la [instalación oficial](https://pytorch.org/get-started/previous-versions/). Si ya tienes un entorno funcional, utiliza su Python sin reinstalarlo.

Comprueba CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"
```

## Primera prueba con una imagen incluida

Este ejemplo selecciona una imagen de validación y ejecuta el peso publicado. No entrena:

```powershell
$Imagen = Get-ChildItem .\datos\sewer-defects\valid\images -File | Sort-Object Name | Select-Object -First 1
.\.venv\Scripts\python.exe .\predecir.py --entrada $Imagen.FullName --confianza 0.25
```

La consola muestra el número de detecciones y la carpeta de salida. Las imágenes anotadas quedan en `resultados/prediccion*`. Se utiliza `pesos/prueba_02.pt` por defecto.

También puedes pasar una imagen propia o una carpeta:

```powershell
.\.venv\Scripts\python.exe .\predecir.py --entrada datos/mi_imagen.jpg --pesos pesos/prueba_02.pt --confianza 0.25
```

`datos/mi_imagen.jpg` es una ruta de ejemplo: debes colocar allí tu imagen o sustituirla por otra ruta existente.

## Los cuatro módulos

| Archivo | Qué hace | Entrada y salida |
| --- | --- | --- |
| [predecir.py](predecir.py) | Carga el peso y ejecuta el motor YOLO11 Nano mediante Ultralytics/PyTorch. | Imagen o carpeta → clases, cajas y confianza; guarda imágenes anotadas. |
| [ver_etiquetas.py](ver_etiquetas.py) | Lee las cajas de los TXT y las dibuja sobre una copia. No ejecuta el detector. | Dataset original → dibujos de las primeras diez imágenes de train. |
| [comparar.py](comparar.py) | Reutiliza la predicción de predecir.py y el dibujo de ver_etiquetas.py. | Muestra reproducible de valid → etiquetas a la izquierda y predicciones a la derecha. |
| [entrenar.py](entrenar.py) | Entrena una nueva ejecución con train y supervisa con valid. No utiliza test para entrenar. | YAML del dataset → pesos y registros nuevos bajo resultados/. |

### Cómo opera el motor

1. Carga `pesos/prueba_02.pt` y comprueba que sus clases coincidan.
2. Ultralytics prepara la imagen y PyTorch ejecuta la red YOLO11n localmente.
3. El detector devuelve clase, coordenadas de caja y puntuación de confianza.
4. Los módulos dibujan y guardan los resultados en copias.

El motor compartido `predecir_imagen` fija tamaño 640, NMS IoU 0,7, máximo 300 detecciones y `rect=False`; comparar.py lo reutiliza. El comando de predecir.py deja los ajustes no especificados a Ultralytics. La confianza predeterminada del CLI sigue siendo 0,50; los ejemplos muestran explícitamente 0,25 como punto de visualización, sin establecer un umbral operativo definitivo.

### Ver las anotaciones existentes

```powershell
.\.venv\Scripts\python.exe .\ver_etiquetas.py
```

Guarda diez dibujos en `resultados/etiquetas/`. Repetirlo puede reemplazar esos dibujos, pero conserva las imágenes y etiquetas originales.

### Comparar etiquetas y predicciones

```powershell
.\.venv\Scripts\python.exe .\comparar.py --cantidad 10 --semilla 20261026 --confianza 0.25
```

Guarda paneles en una subcarpeta nueva de `resultados/comparaciones/`. No abre ventanas salvo que añadas `--abrir`. Mantener semilla, cantidad, datos y pesos permite repetir la misma muestra.

### Entrenar de nuevo, solo cuando lo necesites

```powershell
.\.venv\Scripts\python.exe .\entrenar.py --epocas 30 --nombre entrenamiento_01 --semilla 0
```

Este comando **sí inicia un entrenamiento nuevo**. Parte de YOLO11n preentrenado para objetos generales; no continúa automáticamente desde el peso de defectos publicado. Si la base no está disponible localmente, Ultralytics la descarga dentro de `resultados/pesos_base/`. La base no se distribuye como un segundo peso en GitHub.

Se fijan tamaño 640, batch 4, GPU 0, workers 0 y AdamW con lr0=0,000909. La semilla se registra en la configuración de la ejecución. Usa otro `--nombre` si ya existe una ejecución: el script evita sobrescribirla. Los resultados y nuevos pesos permanecen locales y están excluidos de Git.

Para consultar los parámetros disponibles:

```powershell
.\.venv\Scripts\python.exe .\predecir.py --help
.\.venv\Scripts\python.exe .\comparar.py --help
.\.venv\Scripts\python.exe .\entrenar.py --help
```

## Clases y etiquetas

| ID | Nombre |
| --- | --- |
| 0 | Buckling |
| 1 | Crack |
| 2 | Debris |
| 3 | Hole |
| 4 | Joint offset |
| 5 | Obstacle |
| 6 | Utility intrusion |

Conserva este orden. Cada imagen anotada tiene un TXT del mismo nombre base. Cada fila contiene:

```text
clase centro_x centro_y ancho alto
```

Las coordenadas son proporciones entre 0 y 1. Una imagen puede tener varias cajas y clases; el solape no demuestra un error de anotación.

## Estructura publicada

```text
deteccion-ductos-yolo11/
├── README.md
├── requirements.txt
├── predecir.py
├── ver_etiquetas.py
├── comparar.py
├── entrenar.py
├── pesos/
│   └── prueba_02.pt
└── datos/sewer-defects/
    ├── data.yaml
    ├── README.roboflow.txt
    ├── train/{images,labels}/
    ├── valid/{images,labels}/
    └── test/{images,labels}/
```

`resultados/`, los scripts de experimentación, los otros pesos, las cachés y las imágenes externas quedan fuera de la versión básica. No se borran de la PC del desarrollador al retirarlos del seguimiento.

## Procedencia y límites

Se conserva la exportación **Sewer Defects, versión 4**, de Roboflow, con sus nombres, etiquetas y particiones originales. [Metadatos de la fuente](datos/sewer-defects/data.yaml) y [descripción de la exportación](datos/sewer-defects/README.roboflow.txt). El YAML declara **CC BY 4.0** y enlaza el proyecto de origen; esa declaración del dataset no asigna automáticamente una licencia al código o a las dependencias.

El único checkpoint distribuido es una copia depurada de prueba_02: conserva los parámetros entrenados y las siete clases, sin rutas personales ni resultados históricos del entrenamiento. SHA256:

```text
4006b55b4892df8ab9e200f184bff16b17b729b4d31a8a9b566a4b40982d140d
```

Las predicciones requieren revisión: la confianza no es gravedad ni probabilidad calibrada, y cero detecciones no confirma ausencia de defectos. No se ha certificado independencia por inspección entre las particiones originales; algunas imágenes de test se exploraron anteriormente. Mantén test fuera del ajuste de parámetros y no interpretes las pruebas visuales como una validación de uso en campo.
