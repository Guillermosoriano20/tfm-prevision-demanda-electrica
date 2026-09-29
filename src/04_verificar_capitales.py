import math
import time
import requests
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ruta = RAIZ / "referencia" / "capitales.csv"

cap = pd.read_csv(ruta, dtype={"cod_provincia": str}, encoding="utf-8")

print("Filas:", len(cap))
print("Códigos duplicados:", cap["cod_provincia"].duplicated().sum())
print("Valores nulos:", cap.isna().sum().sum())

no_peninsulares = {"07", "35", "38", "51", "52"}
print("Códigos no peninsulares:", sorted(set(cap["cod_provincia"]) & no_peninsulares))

fuera = cap[~cap["lat"].between(35.9, 43.9) | ~cap["lon"].between(-9.4, 3.4)]
print("Fuera de la península:", len(fuera))


def distancia_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))

URL = "https://geocoding-api.open-meteo.com/v1/search"
print("\nComparación con el geocodificador de Open-Meteo:")
for fila in cap.itertuples():
    params = {"name": fila.capital, "countryCode": "ES", "count": 1, "language": "es"}
    res = requests.get(URL, params=params, timeout=30).json().get("results")
    if not res:
        print(f"  {fila.capital}: no encontrada")
    else:
        d = distancia_km(fila.lat, fila.lon, res[0]["latitude"], res[0]["longitude"])
        if d > 10:
            print(f"  {fila.capital}: {d:.1f} km -> {res[0]['name']} ({res[0].get('admin2')})")
    time.sleep(0.5)
print("Fin de la comparación.")