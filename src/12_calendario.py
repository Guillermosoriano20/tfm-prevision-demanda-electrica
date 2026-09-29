from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
REF = RAIZ / "referencia"

ccaa = pd.read_csv(REF / "provincias_ccaa.csv", dtype=str)
pesos = pd.read_csv(REF / "pesos_poblacion_2021.csv", dtype={"cod_provincia": str})
peso_ccaa = ccaa.merge(pesos, on="cod_provincia").groupby("cod_ccaa")["peso"].sum()

fest = pd.read_csv(REF / "festivos_2017_2024.csv", dtype=str)
fest["ccaa"] = fest["ccaa"].replace("todas", ";".join(peso_ccaa.index)).str.split(";")
largo = fest.explode("ccaa").drop_duplicates(["fecha", "ccaa"])
largo["peso"] = largo["ccaa"].map(peso_ccaa)
assert largo["peso"].notna().all(), "Hay comunidades sin peso"

frac = largo.groupby("fecha")["peso"].sum()
frac.index = pd.to_datetime(frac.index)

cal = pd.DataFrame({"fecha": pd.date_range("2017-01-01", "2024-12-31", freq="D")})
cal["dia_semana"] = cal["fecha"].dt.dayofweek
cal["fraccion_festivo"] = cal["fecha"].map(frac).fillna(0.0)

print("Días:", len(cal), "| días con algún festivo:", (cal["fraccion_festivo"] > 0).sum())
print("Mínimo y máximo:", cal["fraccion_festivo"].min(), cal["fraccion_festivo"].max().round(6))
print("Días con fracción >= 0,5:", (cal["fraccion_festivo"] >= 0.5).sum())

cal["festivo"] = cal["fraccion_festivo"] >= 0.5
no_laborable = cal["festivo"] | (cal["dia_semana"] >= 5)
cal["puente"] = (~no_laborable & no_laborable.shift(1, fill_value=False)
                 & no_laborable.shift(-1, fill_value=False))

cal["tipo_dia"] = "laborable"
cal.loc[cal["dia_semana"] >= 5, "tipo_dia"] = "fin_de_semana"
cal.loc[cal["puente"], "tipo_dia"] = "puente"
cal.loc[cal["festivo"], "tipo_dia"] = "festivo"

cal.to_parquet(RAIZ / "data" / "interim" / "calendario.parquet", index=False)

print(cal["tipo_dia"].value_counts())
print("Puentes:", cal.loc[cal["puente"], "fecha"].dt.date.astype(str).tolist())