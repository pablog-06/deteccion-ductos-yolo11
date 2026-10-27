"""Ejecuta el detector de defectos sobre una imagen o carpeta local."""
from pathlib import Path
import argparse
from ultralytics import YOLO

# __file__ es este archivo. Su carpeta será la raíz del proyecto.
PROYECTO = Path(__file__).resolve().parent
PESOS = "pesos/prueba_02.pt"


def cargar_modelo(pesos, clases=None):
    """Carga pesos locales y comprueba el orden de clases si se solicita."""
    ruta = PROYECTO / pesos
    if not ruta.is_file():
        raise FileNotFoundError(f"No existen los pesos: {ruta}")
    modelo = YOLO(str(ruta))
    if clases is not None and modelo.names != dict(enumerate(clases)):
        raise ValueError("Las clases del modelo y del dataset no coinciden.")
    return modelo


def predecir_imagen(modelo, imagen, confianza=0.25, dispositivo=0):
    """Motor compartido sin guardar. Fija tamaño y NMS para comparar."""
    return modelo.predict(
        source=imagen, conf=confianza, device=dispositivo, imgsz=640,
        iou=0.7, max_det=300, rect=False, save=False, verbose=False,
    )[0]


def predecir(entrada, pesos, confianza):
    # Construimos las rutas completas.
    ruta_entrada = PROYECTO / entrada
    ruta_pesos = PROYECTO / pesos

    # Verificamos que los archivos existan antes de continuar.
    if not ruta_entrada.exists():
        raise FileNotFoundError(f"No existe: {ruta_entrada}")

    if not ruta_pesos.is_file():
        raise FileNotFoundError(f"No existen los pesos: {ruta_pesos}")

    # Cargamos la red y sus pesos aprendidos.
    # El peso publicado debe corresponder a las siete clases del dataset incluido.
    from ver_etiquetas import CLASES
    modelo = cargar_modelo(ruta_pesos, CLASES)

    # Ejecutamos la predicción.
    resultados = modelo.predict(
        source=str(ruta_entrada),        # Imagen o carpeta que analizaremos.
        device=0,                       # Primera GPU NVIDIA.
        conf=confianza,                  # Umbral para mostrar detecciones.
        save=True,                      # Guarda imágenes con cajas.
        project=str(PROYECTO / "resultados"),
        name="prediccion",
        stream=True,                    # Entrega un resultado por vez.
    )

    # Recorremos los resultados para procesar todas las imágenes.
    for resultado in resultados:
        cantidad = len(resultado.boxes)
        print(f"{Path(resultado.path).name}: {cantidad} detecciones")
        print("Guardado en:", resultado.save_dir)


# Este bloque solo se ejecuta al iniciar el archivo desde la terminal.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predecir con YOLO")

    # Argumentos que escribiremos después de python predecir.py.
    parser.add_argument("--entrada", required=True)
    parser.add_argument("--pesos", default=PESOS)
    parser.add_argument("--confianza", type=float, default=0.50)

    args = parser.parse_args()

    if not 0 <= args.confianza <= 1:
        parser.error("La confianza debe estar entre 0 y 1.")

    predecir(args.entrada, args.pesos, args.confianza)
