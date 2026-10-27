"""Compara etiquetas y predicciones con una selección repetible."""
from pathlib import Path
import argparse
import csv
import hashlib
import os
import random

import cv2
import yaml

from predecir import cargar_modelo, predecir_imagen
from ver_etiquetas import dibujar_etiquetas

PROYECTO = Path(__file__).resolve().parent
DATOS = PROYECTO / "datos" / "sewer-defects"
PESOS = "pesos/prueba_02.pt"


def agregar_titulo(imagen, texto):
    copia = cv2.copyMakeBorder(
        imagen, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30),
    )
    cv2.putText(copia, texto, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)
    return copia


def guardar_comparacion(etiqueta, clases, resultado, destino, confianza):
    """Acepta un resultado calculado: no repite la inferencia."""
    izquierda = dibujar_etiquetas(resultado.orig_img, etiqueta, clases)
    izquierda = agregar_titulo(izquierda, "ETIQUETAS | G = fila del TXT (desde 0)")
    derecha = agregar_titulo(resultado.plot(), f"PREDICCIONES | conf={confianza}")
    comparacion = cv2.hconcat([izquierda, derecha])
    destino = Path(destino)
    if not cv2.imwrite(str(destino), comparacion):
        raise OSError(f"No se pudo guardar: {destino}")
    return destino


def leer_muestra(manifiesto):
    """Los hashes evitan comparar otra versión del dataset sin advertirlo."""
    with Path(manifiesto).open(encoding="utf-8-sig", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    rutas = []
    for fila in filas:
        ruta = PROYECTO / fila["imagen"]
        # Un manifiesto nuevo puede apuntar a etiquetas derivadas, no a valid original.
        etiqueta = (PROYECTO / fila["etiqueta"] if fila.get("etiqueta") else
                    ruta.parent.parent / "labels" / f"{ruta.stem}.txt")
        for nombre, archivo in [("sha256_imagen", ruta), ("sha256_etiqueta", etiqueta)]:
            if hashlib.sha256(archivo.read_bytes()).hexdigest() != fila[nombre]:
                raise ValueError(f"Cambió un archivo de la muestra: {archivo}")
        rutas.append(ruta)
    return rutas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pesos", default=PESOS)
    parser.add_argument("--manifiesto", help="CSV muestra_valid.csv generado por el análisis")
    parser.add_argument("--cantidad", type=int, default=10)
    parser.add_argument("--semilla", type=int, default=20261026)
    parser.add_argument("--confianza", type=float, default=0.25)
    parser.add_argument("--salida", default="resultados/comparaciones")
    parser.add_argument("--abrir", action="store_true", help="Abre cada imagen; por defecto solo guarda")
    args = parser.parse_args()
    if not 0 <= args.confianza <= 1 or args.cantidad < 1:
        parser.error("Confianza entre 0 y 1 y cantidad positiva.")
    with (DATOS / "data.yaml").open(encoding="utf-8") as archivo:
        clases = yaml.safe_load(archivo)["names"]
    modelo = cargar_modelo(args.pesos, clases)
    etiquetas_manifiesto = {}
    if args.manifiesto:
        seleccionadas = leer_muestra(PROYECTO / args.manifiesto)
        with (PROYECTO / args.manifiesto).open(encoding="utf-8-sig", newline="") as archivo:
            etiquetas_manifiesto = {
                PROYECTO / fila["imagen"]: PROYECTO / fila["etiqueta"]
                for fila in csv.DictReader(archivo) if fila.get("etiqueta")
            }
    else:
        imagenes = sorted(
            ruta for ruta in (DATOS / "valid" / "images").iterdir()
            if ruta.is_file() and ruta.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        seleccionadas = random.Random(args.semilla).sample(imagenes, min(args.cantidad, len(imagenes)))
    # Una subcarpeta nueva conserva las comparaciones anteriores.
    base = PROYECTO / args.salida
    if not base.resolve().is_relative_to((PROYECTO / "resultados").resolve()):
        parser.error("La salida debe estar dentro de resultados/.")
    salida = base / "muestra_01"
    numero = 1
    while salida.exists():
        numero += 1
        salida = base / f"muestra_{numero:02d}"
    salida.mkdir(parents=True)
    for numero, ruta in enumerate(seleccionadas, 1):
        resultado = predecir_imagen(modelo, str(ruta), args.confianza)
        destino = guardar_comparacion(
            etiquetas_manifiesto.get(ruta, ruta.parent.parent / "labels" / f"{ruta.stem}.txt"), clases,
            resultado, salida / f"{numero:02d}_{ruta.stem}.jpg", args.confianza,
        )
        if args.abrir:
            os.startfile(str(destino))
    print(f"Guardadas {len(seleccionadas)} comparaciones en {salida}")


if __name__ == "__main__":
    main()
