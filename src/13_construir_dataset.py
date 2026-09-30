"""Construcción del dataset horario: demanda, temperatura y calendario."""
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DIR_ESIOS = RAIZ / "data" / "raw" / "esios_1293"

# --- Fragmento 1: lectura de la demanda cruda ---
ficheros = sorted(DIR_ESIOS.glob("*.parquet"))
print("Ficheros:", len(ficheros))
crudo = pd.concat([pd.read_parquet(f) for f in ficheros], ignore_index=True)
print("Filas:", len(crudo))
print(crudo.dtypes)
print(crudo.head(3))

# --- Fragmento 2: demanda media horaria (UTC) ---
# Convención: la hora H agrupa los valores de H:00 a H:55 (etiqueta = inicio).
print("geo_id:", crudo["geo_id"].unique())
crudo["ts_utc"] = pd.to_datetime(crudo["datetime_utc"], utc=True)
crudo["hora_utc"] = crudo["ts_utc"].dt.floor("h")
demanda = crudo.groupby("hora_utc")["value"].agg(demanda_mw="mean", n_valores="size")
print("Horas:", len(demanda))
print(demanda["n_valores"].value_counts().sort_index())
print(demanda.head(3))
print(demanda.tail(3))

# --- Fragmento 3: hora peninsular (Europe/Madrid) ---
demanda = demanda.reset_index()
demanda["hora_local"] = demanda["hora_utc"].dt.tz_convert("Europe/Madrid")
demanda["fecha_local"] = demanda["hora_local"].dt.date
demanda["hora_del_dia"] = demanda["hora_local"].dt.hour
horas_por_dia = demanda.groupby("fecha_local").size()
print(horas_por_dia.value_counts().sort_index())
print(horas_por_dia[horas_por_dia != 24])

# --- Fragmento 4: temperatura (inspección) ---
DIR_INTERIM = RAIZ / "data" / "interim"
temp = pd.read_parquet(DIR_INTERIM / "temperatura_provincias.parquet")
print(temp.shape)
print(temp.index[:3])
print(temp.index[-3:])
print(temp.columns[:5].tolist(), temp.columns[-3:].tolist())
print(temp.dtypes.value_counts())

# --- Fragmento 5: alineación trapezoidal y unión con la demanda ---
assert (temp.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
temp_hora = (temp + temp.shift(-1)) / 2  # hora H = media de T(H) y T(H+1)
temp_hora.columns = [c if c == "t_ponderada" else f"t_{c}" for c in temp_hora.columns]
demanda["hora_utc"] = demanda["hora_utc"].dt.as_unit("us")
datos = demanda.merge(temp_hora, left_on="hora_utc", right_index=True,
                      how="left", validate="one_to_one")
print(datos.shape)
print(datos.isna().sum().value_counts())
print(datos.loc[datos["t_ponderada"].isna(), ["hora_utc", "hora_local"]])
print(datos["t_ponderada"].describe().round(2))

# --- Fragmento 6: calendario (inspección) ---
cal = pd.read_parquet(DIR_INTERIM / "calendario.parquet")
print(cal.shape)
print(cal.dtypes)
print(cal.head(3))
print(cal["tipo_dia"].value_counts())

# --- Fragmento 7: unión con el calendario por fecha local ---
datos["fecha_local"] = pd.to_datetime(datos["fecha_local"]).dt.as_unit("us")
datos = datos.merge(cal, left_on="fecha_local", right_on="fecha",
                    how="left", validate="many_to_one").drop(columns="fecha")
print(datos.shape)
print("Horas sin calendario:", datos["tipo_dia"].isna().sum())
ok = datos["dia_semana"] == datos["hora_local"].dt.dayofweek
print("dia_semana coincide:", ok.sum(), "de", len(datos))

# --- Fragmento 8: última hora de 2025 y marca del periodo COVID ---
datos = datos[datos["fecha_local"] < "2025-01-01"].reset_index(drop=True)
datos = datos.astype({"festivo": bool, "puente": bool, "dia_semana": "int32"})
datos["excluir_covid"] = datos["fecha_local"].between("2020-03-01", "2020-06-30")
print(datos.shape)
print("Nulos totales:", datos.isna().sum().sum())
print("Horas COVID:", datos["excluir_covid"].sum())
print(datos.loc[datos["excluir_covid"], "fecha_local"].agg(["min", "max"]))

# --- Fragmento 9: comprobaciones finales y guardado ---
assert len(datos) == 70127 and datos.isna().sum().sum() == 0
assert (datos["hora_utc"].diff().dropna() == pd.Timedelta("1h")).all()
assert (datos["demanda_mw"] > 0).all()
assert datos["n_valores"].isin([6, 12]).all()
print(datos["hora_local"].iloc[[0, -1]].tolist())
print(datos["demanda_mw"].describe().round(0))
SALIDA = RAIZ / "data" / "processed" / "dataset.parquet"
SALIDA.parent.mkdir(exist_ok=True)
datos.to_parquet(SALIDA, index=False)
print("Relectura idéntica:", pd.read_parquet(SALIDA).equals(datos))
print("Huella:", pd.util.hash_pandas_object(datos, index=False).sum())

print(datos.loc[datos["demanda_mw"].idxmin(), ["hora_local", "demanda_mw", "tipo_dia", "excluir_covid"]])
print("Mínimo fuera de COVID:", datos.loc[~datos["excluir_covid"], "demanda_mw"].min().round(0))