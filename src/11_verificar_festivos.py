from pathlib import Path
import pandas as pd
import holidays

RAIZ = Path(__file__).resolve().parents[1]
REF = RAIZ / "referencia"
fest = pd.read_csv(REF / "festivos_2017_2024.csv", dtype=str, encoding="utf-8")
ccaa = sorted(pd.read_csv(REF / "provincias_ccaa.csv", dtype=str)["cod_ccaa"].unique())

fest["ccaa"] = fest["ccaa"].replace("todas", ";".join(ccaa)).str.split(";")
largo = fest.explode("ccaa")
largo["anio"] = largo["fecha"].str[:4]

print("Filas del CSV:", len(fest), "| nulos:", fest.isna().sum().sum())
print("Códigos desconocidos:", sorted(set(largo["ccaa"]) - set(ccaa)))
print("Fecha y comunidad repetidas:", largo.duplicated(["fecha", "ccaa"]).sum())
print("Fechas no válidas:", pd.to_datetime(fest["fecha"], errors="coerce").isna().sum())

conteo = largo.groupby(["anio", "ccaa"]).size()
print("Recuentos distintos de 12:")
print(conteo[conteo != 12])

SUBDIV = {"01": "AN", "02": "AR", "03": "AS", "06": "CB", "07": "CL", "08": "CM",
          "09": "CT", "10": "VC", "11": "EX", "12": "GA", "13": "MD", "14": "MC",
          "15": "NC", "16": "PV", "17": "RI"}

nuestras = set(largo["ccaa"] + " " + largo["fecha"])
libreria = set()
for cod, sub in SUBDIV.items():
    for dia in holidays.Spain(subdiv=sub, years=range(2017, 2025)):
        if dia.weekday() != 6:  # 6 = domingo
            libreria.add(f"{cod} {dia}")

print("\nSolo en nuestro CSV:", len(nuestras - libreria))
print(sorted(nuestras - libreria))
print("Solo en la librería:", len(libreria - nuestras))
print(sorted(libreria - nuestras))