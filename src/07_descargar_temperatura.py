from pathlib import Path
import time
import requests
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "data" / "raw" / "openmeteo_era5land"
SALIDA.mkdir(parents=True, exist_ok=True)
URL = "https://archive-api.open-meteo.com/v1/archive"
LIMITE = None 

cap = pd.read_csv(RAIZ / "referencia" / "capitales.csv", dtype={"cod_provincia": str})
filas = cap.iloc[:LIMITE]
print("Capitales a descargar:", len(filas), "->", filas["capital"].tolist())

for fila in filas.itertuples():
    destino = SALIDA / f"{fila.cod_provincia}.parquet"
    if destino.exists():
        print(fila.capital, "ya descargada")
        continue
    params = {"latitude": fila.lat, "longitude": fila.lon,
              "start_date": "2017-01-01", "end_date": "2024-12-31",
              "hourly": "temperature_2m", "models": "era5_land", "timezone": "GMT"}
    r = requests.get(URL, params=params, timeout=120)
    if not r.ok:
        print("Error en", fila.capital, r.status_code, r.text)
        break
    d = r.json()
    df = pd.DataFrame({"hora_utc": pd.to_datetime(d["hourly"]["time"], utc=True),
                       "temperatura": d["hourly"]["temperature_2m"]})
    df["cod_provincia"] = fila.cod_provincia
    df["celda_lat"], df["celda_lon"] = d["latitude"], d["longitude"]
    df.to_parquet(destino, index=False)
    print(fila.capital, len(df), "horas | nulos:", df["temperatura"].isna().sum(),
          "| celda:", d["latitude"], d["longitude"])
    time.sleep(2)