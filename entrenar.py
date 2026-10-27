"""Entrena YOLO11 Nano con train y supervisa con valid, sin modificar los datos."""
from pathlib import Path
import argparse
import torch
import yaml
from ultralytics import YOLO
from ver_etiquetas import CLASES

PROYECTO = Path(__file__).resolve().parent


def entrenar(datos=None, epocas=30, nombre="entrenamiento_01", semilla=0):
    ruta_datos = PROYECTO / (datos or "datos/sewer-defects/data.yaml")
    if not ruta_datos.is_file():
        raise FileNotFoundError(ruta_datos)
    if not torch.cuda.is_available():
        raise RuntimeError("Este entrenamiento necesita la GPU CUDA local.")
    config_datos = yaml.safe_load(ruta_datos.read_text(encoding="utf-8"))
    if config_datos.get("names") != CLASES:
        raise ValueError("El dataset debe conservar las siete clases y su orden.")
    proyecto = PROYECTO / "resultados" / "entrenamiento"
    if (proyecto / nombre).exists():
        raise FileExistsError("Ese entrenamiento ya existe; utiliza otro --nombre.")
    # El repositorio distribuye un solo peso de defectos. La base para volver a
    # entrenar se obtiene aparte, dentro de resultados/, cuando sea necesaria.
    base = PROYECTO / "yolo11n.pt"
    if not base.is_file():
        base = PROYECTO / "resultados" / "pesos_base" / "yolo11n.pt"
        base.parent.mkdir(parents=True, exist_ok=True)
    modelo = YOLO(str(base))
    configuracion = {
        "data": str(ruta_datos), "epochs": epocas, "imgsz": 640, "batch": 4,
        "device": 0, "workers": 0, "name": nombre,
        "project": str(proyecto), "seed": semilla, "deterministic": True,
        "optimizer": "AdamW", "lr0": 0.000909, "momentum": 0.9,
        "weight_decay": 0.0005, "warmup_bias_lr": 0.0, "warmup_epochs": 3.0,
        "lrf": 0.01, "cos_lr": False, "close_mosaic": 10,
        "patience": 20, "resume": False,
    }
    return modelo.train(**configuracion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", help="YAML del dataset, relativo al proyecto")
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--nombre", default="entrenamiento_01")
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()
    if args.epocas < 1 or Path(args.nombre).name != args.nombre:
        parser.error("Épocas positivas y nombre sin directorios.")
    entrenar(args.datos, args.epocas, args.nombre, args.semilla)
