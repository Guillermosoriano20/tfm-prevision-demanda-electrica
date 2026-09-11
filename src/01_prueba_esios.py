import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("ESIOS_TOKEN")

url = "https://api.esios.ree.es/indicators/1293"
cabeceras = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "x-api-key": token,
}
parametros = {
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-31T23:59:59Z",
}

respuesta = requests.get(url, headers=cabeceras, params=parametros, timeout=60)
print("Código de estado:", respuesta.status_code)

if respuesta.status_code == 200:
    valores = respuesta.json()["indicator"]["values"]
    print("Número de valores recibidos:", len(valores))
    df = pd.DataFrame(valores)
    print("Columnas:", df.columns.tolist())
    print("Primer instante:", df["datetime_utc"].min(), "| Último:", df["datetime_utc"].max())
    if "geo_name" in df.columns:
        print(df["geo_name"].value_counts())
    print(df.head(6).to_string())
else:
    print("Mensaje del servidor:")
    print(respuesta.text[:300])