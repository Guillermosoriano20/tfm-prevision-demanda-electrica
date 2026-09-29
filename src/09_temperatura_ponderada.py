from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "data" / "raw" / "openmeteo_era5land"

partes = [pd.read_parquet(f) for f in sorted(ENTRADA.glob("*.parquet"))]
largo = pd.concat(partes, ignore_index=True)
ancho = largo.pivot(index="hora_utc", columns="cod_provincia", values="temperatura")

print("Forma:", ancho.shape)
print("Nulos:", ancho.isna().sum().sum())
print("Desde", ancho.index.min(), "hasta", ancho.index.max())

pesos = pd.read_csv(RAIZ / "referencia" / "pesos_poblacion_2021.csv",
                    dtype={"cod_provincia": str}).set_index("cod_provincia")["peso"]
pesos = pesos.reindex(ancho.columns)
assert pesos.notna().all(), "Hay provincias sin peso"

ancho["t_ponderada"] = ancho[pesos.index].mul(pesos, axis=1).sum(axis=1)

SALIDA = RAIZ / "data" / "interim"
SALIDA.mkdir(parents=True, exist_ok=True)
ancho.to_parquet(SALIDA / "temperatura_provincias.parquet")

print(ancho["t_ponderada"].describe().round(1))