"""Visualiza etiquetas existentes. Importar este archivo no crea salidas."""
from pathlib import Path
import cv2
import yaml

PROYECTO = Path(__file__).resolve().parent
DATOS = PROYECTO / "datos" / "sewer-defects"
SALIDA = PROYECTO / "resultados" / "etiquetas"
CANTIDAD = 10
CLASES = ["Buckling", "Crack", "Debris", "Hole", "Joint offset", "Obstacle", "Utility intrusion"]


def leer_etiquetas(ruta):
    """Filas [clase, centro_x, centro_y, ancho, alto] normalizadas."""
    filas = []
    for numero, linea in enumerate(Path(ruta).read_text(encoding="utf-8").splitlines(), 1):
        if not linea.strip():
            continue
        valores = linea.split()
        if len(valores) != 5:
            raise ValueError(f"Etiqueta inválida en {ruta}, línea {numero}")
        filas.append([int(valores[0]), *map(float, valores[1:])])
    return filas


def dibujar_etiquetas(imagen, etiqueta, clases):
    """Dibuja sobre una copia; G0, G1... identifican las filas del TXT."""
    copia = imagen.copy()
    alto, ancho = copia.shape[:2]
    for indice, (clase, cx, cy, w, h) in enumerate(leer_etiquetas(etiqueta)):
        x1, y1 = round((cx - w / 2) * ancho), round((cy - h / 2) * alto)
        x2, y2 = round((cx + w / 2) * ancho), round((cy + h / 2) * alto)
        cv2.rectangle(copia, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            copia, f"G{indice} {clases[clase]}", (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
    return copia


def main():
    with (DATOS / "data.yaml").open(encoding="utf-8") as archivo:
        clases = yaml.safe_load(archivo)["names"]
    SALIDA.mkdir(parents=True, exist_ok=True)
    imagenes = sorted(
        ruta for ruta in (DATOS / "train" / "images").iterdir()
        if ruta.is_file() and ruta.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for ruta in imagenes[:CANTIDAD]:
        etiqueta = DATOS / "train" / "labels" / f"{ruta.stem}.txt"
        if not etiqueta.is_file():
            print("Falta la etiqueta de:", ruta.name)
            continue
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            print("No se pudo leer:", ruta.name)
            continue
        destino = SALIDA / ruta.name
        if not cv2.imwrite(str(destino), dibujar_etiquetas(imagen, etiqueta, clases)):
            raise OSError(f"No se pudo guardar: {destino}")
        print("Guardada:", destino.name)
    print("Resultados en:", SALIDA)


if __name__ == "__main__":
    main()
