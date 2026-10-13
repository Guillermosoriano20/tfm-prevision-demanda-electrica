"""LightGBM: comprobaciones de las variables y prueba de fuga (paso L3).

Variables: docs/protocolo_oe5.md, apartado 11.1. Se ajustan con los datos hasta el
31/12/2022 12:00 tres modelos de referencia con los hiperparámetros por defecto de
LightGBM (puntual y cuantiles 0,05 y 0,95, apartado 11.2) y se pasa la prueba de fuga
en los 365 orígenes de 2023. No se calcula ningún error de 2023: solo se comparan
previsiones con la demanda futura alterada. Control positivo: la misma prueba con un
retardo de 24 h colado en la variable y_48 debe detectar fuga.
"""
import time

import pandas as pd

import validacion as v
from lightgbm_modelo import (COLUMNAS, ajustar, filas_entrenamiento, hacer_prever,
                             variables)

datos = v.cargar_datos()                                   # huella comprobada; 2024 cortado
origenes = v.generar_origenes("validacion")
fin = origenes["origen_utc"].iloc[0]                       # 31/12/2022 12:00 local

# --- 1. Filas de entrenamiento (11.1.6 y 11.1.7) ---
inicio = time.perf_counter()
X, y = filas_entrenamiento(datos, fin)
local = X.index.tz_convert(v.ZONA)
print(f"Filas de entrenamiento: {len(X)} x {X.shape[1]} variables"
      f" ({time.perf_counter() - inicio:.1f} s)")
print(f"  primera hora objetivo: {local[0]:%d/%m/%Y %H:%M}, última: {local[-1]:%d/%m/%Y %H:%M}")
saltos = (X.index[1:] - X.index[:-1]) > v.UNA_HORA        # horas excluidas en medio
for i in saltos.nonzero()[0]:
    print(f"  excluidas entre {local[i]:%d/%m/%Y %H:%M} y {local[i + 1]:%d/%m/%Y %H:%M}")
print(f"  anticipación: de {X['anticipacion'].min():.0f} a {X['anticipacion'].max():.0f} h")
assert list(X.columns) == COLUMNAS and not X.isna().any().any()

# --- 2. Modelos de referencia (hiperparámetros por defecto, 100 árboles) ---
modelos = {}
for nombre, alpha in [("puntual", None), ("cuantil 0,05", 0.05), ("cuantil 0,95", 0.95)]:
    inicio = time.perf_counter()
    modelos[nombre] = ajustar(X, y, {}, alpha=alpha)
    print(f"Ajuste {nombre}: {time.perf_counter() - inicio:.1f} s")

# --- 3. Prueba de fuga en los 365 orígenes de 2023 ---
for nombre, modelo in modelos.items():
    inicio = time.perf_counter()
    fugas = v.comprobar_fuga(hacer_prever(modelo), datos, origenes)
    print(f"Prueba de fuga, {nombre}: {len(fugas)} días con fuga {fugas[:3]}"
          f" ({time.perf_counter() - inicio:.0f} s)", flush=True)
    assert not fugas, "FUGA DETECTADA: se detiene todo"

# --- 4. Control positivo: retardo de 24 h colado en y_48 (debe dar fuga) ---
def prever_con_fuga(datos, origen):
    horas = pd.date_range(origen.inicio_utc, origen.fin_utc, freq="h",
                          inclusive="left").as_unit(datos.index.unit)
    Xo = variables(datos, horas, pd.DatetimeIndex([origen.origen_utc] * len(horas)))
    Xo["y_48"] = datos["demanda_mw"].reindex(horas - 24 * v.UNA_HORA).to_numpy()
    return pd.Series(modelos["puntual"].predict(Xo), index=horas)


fugas = v.comprobar_fuga(prever_con_fuga, datos, origenes)
print(f"Control positivo (retardo de 24 h): {len(fugas)} días con fuga {fugas[:3]}")
