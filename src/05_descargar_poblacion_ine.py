from pathlib import Path
import requests
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
destino = RAIZ / "data" / "raw" / "ine_2852.csv"
URL = "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/2852.csv"

r = requests.get(URL, timeout=60)
r.raise_for_status()
destino.write_bytes(r.content)

pob = pd.read_csv(destino, sep=";", encoding="utf-8-sig", dtype=str)
print(pob.shape)
print(pob.columns.tolist())
print(pob.head(8))

cap = pd.read_csv(RAIZ / "referencia" / "capitales.csv", dtype={"cod_provincia": str})

sel = pob[(pob["Periodo"] == "2021") & (pob["Sexo"] == "Total")].copy()
sel["cod_provincia"] = sel["Provincias"].str.extract(r"^(\d{2})", expand=False)
sel = sel[sel["cod_provincia"].isin(cap["cod_provincia"])]

sel["poblacion_2021"] = sel["Total"].str.replace(".", "", regex=False).astype(int)
sel["peso"] = sel["poblacion_2021"] / sel["poblacion_2021"].sum()

pesos = sel[["cod_provincia", "Provincias", "poblacion_2021", "peso"]].sort_values("cod_provincia")
pesos.to_csv(RAIZ / "referencia" / "pesos_poblacion_2021.csv", index=False, encoding="utf-8")

print("Provincias con peso:", len(pesos))
print("Suma de pesos:", round(pesos["peso"].sum(), 6))
print(pesos.sort_values("peso", ascending=False).head(3))