from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "data" / "raw" / "esios_1293"

archivos = sorted(CARPETA.glob("*.parquet"))
df = pd.concat([pd.read_parquet(a) for a in archivos], ignore_index=True)
print("Archivos leídos:", len(archivos), "| Filas totales:", len(df))

df["instante"] = pd.to_datetime(df["datetime_utc"], utc=True)
print("Instantes duplicados:", df["instante"].duplicated().sum())
print("Valores vacíos:", df["value"].isna().sum())
print("Valores <= 0:", (df["value"] <= 0).sum())

por_dia = df.groupby(df["instante"].dt.date).size()
raros = por_dia[~por_dia.isin([144, 288])]
print("Días con recuento distinto de 144 o 288:", len(raros))
print(raros.to_string())

df = df.sort_values("instante")
saltos = df["instante"].diff().dropna()
print("Distancias entre datos consecutivos:")
print(saltos.value_counts().to_string())

primer_5min = df.loc[saltos[saltos == pd.Timedelta(minutes=5)].index, "instante"].min()
print("Primer dato a 5 minutos:", primer_5min)
print("Primer instante:", df["instante"].min(), "| Último:", df["instante"].max())