from pathlib import Path
import requests
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
URL = "https://archive-api.open-meteo.com/v1/archive"
cap = pd.read_csv(RAIZ / "referencia" / "capitales.csv", dtype={"cod_provincia": str})

comparacion = {}
for cod in ["08", "11"]:
    fila = cap[cap["cod_provincia"] == cod].iloc[0]
    params = {"latitude": fila["lat"], "longitude": fila["lon"],
              "start_date": "2017-01-01", "end_date": "2024-12-31",
              "hourly": "temperature_2m", "models": "era5", "timezone": "GMT",
              "cell_selection": "nearest"}
    r = requests.get(URL, params=params, timeout=120)
    if not r.ok:
        print("Error en", fila["capital"], r.status_code, r.text)
        break
    d = r.json()
    era5 = pd.Series(d["hourly"]["temperature_2m"],
                     index=pd.to_datetime(d["hourly"]["time"], utc=True))
    ruta_land = RAIZ / "data" / "raw" / "openmeteo_era5land" / f"{cod}.parquet"
    land = pd.read_parquet(ruta_land).set_index("hora_utc")["temperatura"]
    comparacion[fila["capital"]] = pd.DataFrame({"era5_land": land, "era5": era5})
    print(fila["capital"], "| celda ERA5:", d["latitude"], d["longitude"],
          "| horas:", len(comparacion[fila["capital"]]))

for capital, df in comparacion.items():
    dif = df["era5_land"] - df["era5"]
    diaria = df.resample("D")
    amp = (diaria.max() - diaria.min()).mean()
    print(f"\n{capital}")
    print("  Correlación horaria:", round(df.corr().iloc[0, 1], 3))
    print("  Correlación medias diarias:", round(diaria.mean().corr().iloc[0, 1], 3))
    print("  Diferencia media (Land - ERA5):", round(dif.mean(), 2), "°C")
    print("  Desv. típica de la diferencia:", round(dif.std(), 2), "°C")
    print("  Amplitud diaria media Land / ERA5:",
          round(amp["era5_land"], 1), "/", round(amp["era5"], 1), "°C")