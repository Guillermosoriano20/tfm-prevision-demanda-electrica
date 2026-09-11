import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("ESIOS_TOKEN")

URL = "https://api.esios.ree.es/indicators/1293"
CABECERAS = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "x-api-key": TOKEN,
}

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "data" / "raw" / "esios_1293"


def descargar_mes(anio, mes):
    inicio = pd.Timestamp(year=anio, month=mes, day=1, tz="UTC")
    fin = inicio + pd.offsets.MonthBegin(1) - pd.Timedelta(seconds=1)
    parametros = {
        "start_date": inicio.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_date": fin.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    respuesta = requests.get(URL, headers=CABECERAS, params=parametros, timeout=120)
    respuesta.raise_for_status()
    df = pd.DataFrame(respuesta.json()["indicator"]["values"])
    if df.empty:
        raise ValueError("ESIOS ha devuelto 0 valores")

    CARPETA.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA / f"{anio}-{mes:02d}.parquet"
    df.to_parquet(ruta, index=False)
    return ruta, len(df)


for anio in range(2017, 2025):
    for mes in range(1, 13):
        ruta = CARPETA / f"{anio}-{mes:02d}.parquet"
        if ruta.exists():
            print("Ya existe, se salta:", ruta.name)
            continue
        try:
            ruta, filas = descargar_mes(anio, mes)
            print("Guardado:", ruta.name, "|", filas, "filas")
        except Exception as error:
            print("ERROR en", anio, mes, "->", error)
        time.sleep(2)