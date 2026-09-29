from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
REF = RAIZ / "referencia"
ccaa = pd.read_csv(REF / "provincias_ccaa.csv", dtype={"cod_provincia": str, "cod_ccaa": str})
pesos = pd.read_csv(REF / "pesos_poblacion_2021.csv", dtype={"cod_provincia": str})

union = ccaa.merge(pesos, on="cod_provincia", how="outer", indicator=True)
print("Filas:", len(union), "| sin emparejar:", (union["_merge"] != "both").sum())

resumen = union.groupby(["cod_ccaa", "ccaa"]).agg(
    provincias=("cod_provincia", "size"), peso=("peso", "sum"))
print(resumen.round(3))
print("Comunidades:", len(resumen), "| suma de pesos:", round(resumen["peso"].sum(), 6))