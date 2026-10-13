"""LightGBM: búsqueda aleatoria de 20 configuraciones evaluadas en 2023 (validación).

Protocolo: docs/protocolo_oe5.md, apartado 11.3 (espacio y procedimiento registrados con
commit antes de lanzar). Variables 11.1; intervalos 11.2. Cada configuración: parada
temprana con las últimas 8 semanas de 2017-2022, reajuste con toda la ventana y
previsión de los 365 orígenes de 2023 sin reajustes. Decide el menor MAE. Las
previsiones de cada configuración se guardan en data/predicciones/: si se interrumpe,
al relanzarlo no se repiten las ya hechas.
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.model_selection import ParameterSampler

import validacion as v
from lightgbm_modelo import (MAX_ARBOLES, PACIENCIA, entrenar, filas_entrenamiento,
                             hacer_prever)

# --- Espacio de búsqueda (11.3): 19 sorteadas con semilla 42 + la referencia ---
ESPACIO = {
    "objective": ["regression", "regression_l1"],
    "learning_rate": loguniform(0.02, 0.2),
    "num_leaves": [15, 31, 63, 127, 255],
    "min_data_in_leaf": [20, 50, 100, 200, 500],
    "feature_fraction": uniform(0.5, 0.5),          # uniforme entre 0,5 y 1
    "bagging_fraction": uniform(0.5, 0.5),
    "lambda_l2": loguniform(1e-3, 10),
}
CONFIGURACIONES = {"c00": {}}                       # c00: valores por defecto de LightGBM
for i, c in enumerate(ParameterSampler(ESPACIO, n_iter=19, random_state=42), 1):
    c = {k: (x if isinstance(x, str) else int(x) if k in ("num_leaves", "min_data_in_leaf")
             else float(x)) for k, x in c.items()}
    CONFIGURACIONES[f"c{i:02d}"] = {**c, "bagging_freq": 1}

# Comprobación: el sorteo reproduce el registrado en el protocolo (learning_rate a 4 cifras)
TASAS_REGISTRADAS = [0.0794, 0.1021, 0.0305, 0.0502, 0.0482, 0.0206, 0.0967, 0.0304, 0.1865,
                     0.0424, 0.1348, 0.0293, 0.0316, 0.1644, 0.0927, 0.1153, 0.0318, 0.0515,
                     0.1519]
tasas = [round(CONFIGURACIONES[f"c{i:02d}"]["learning_rate"], 4) for i in range(1, 20)]
assert tasas == TASAS_REGISTRADAS, "el sorteo no coincide con el registrado"

datos = v.cargar_datos()                                   # huella comprobada; 2024 cortado
origenes = v.generar_origenes("validacion")
fin = origenes["origen_utc"].iloc[0]                       # 31/12/2022 12:00 local
X, y = filas_entrenamiento(datos, fin)
escala = v.escala_mase(datos)
ingenuo = pd.read_parquet(v.DIR_PREDICCIONES / "ingenuo_validacion.parquet")
mae_ingenuo = (ingenuo["real"] - ingenuo["prevision"]).abs().mean()
RESUMEN = v.RAIZ / "data" / "interim" / "lightgbm_busqueda.parquet"
hechas = pd.read_parquet(RESUMEN).set_index("config") if RESUMEN.exists() else None

# --- 1. Las 20 configuraciones ---
filas, modelos = [], {}
for nombre, parametros in CONFIGURACIONES.items():
    archivo = v.DIR_PREDICCIONES / f"lightgbm_{nombre}_validacion.parquet"
    if archivo.exists() and hechas is not None and nombre in hechas.index:
        anterior = hechas.loc[nombre]
        filas.append({"config": nombre, **anterior.to_dict()})
        print(f"  {nombre}: guardada, MAE {anterior['MAE']:.1f} MW", flush=True)
        continue
    inicio = time.perf_counter()
    try:
        modelo, arboles = entrenar(X, y, fin, parametros)
        tabla = v.evaluar(hacer_prever(modelo), datos, "validacion", f"lightgbm_{nombre}")
    except Exception as error:                     # se registra como fallida (11.3)
        print(f"  {nombre}: FALLIDA ({error})", flush=True)
        filas.append({"config": nombre, "estado": "fallida"})
        continue
    m = v.metricas(tabla, escala)
    sesgo = (tabla["real"] - tabla["prevision"]).mean()
    filas.append({"config": nombre, "estado": "valida", **m.to_dict(), "sesgo": sesgo,
                  "MAE_rel_ingenuo": m["MAE"] / mae_ingenuo, "arboles": arboles,
                  "tope": arboles > MAX_ARBOLES - PACIENCIA,
                  "minutos": (time.perf_counter() - inicio) / 60})
    modelos[nombre] = modelo
    pd.DataFrame(filas).to_parquet(RESUMEN, index=False)
    print(f"  {nombre}: MAE {m['MAE']:.1f} MW, sesgo {sesgo:.0f} MW, {arboles} árboles"
          f" ({filas[-1]['minutos']:.1f} min)", flush=True)

resumen = pd.DataFrame(filas)
resumen.to_parquet(RESUMEN, index=False)
validas = resumen[resumen["estado"] == "valida"].sort_values("MAE")
parametros = pd.DataFrame.from_dict(CONFIGURACIONES, orient="index")
parametros.loc["c00", ["objective", "learning_rate", "num_leaves", "min_data_in_leaf"]] = [
    "regression", 0.1, 31, 20]                      # valores por defecto, solo para la tabla
parametros[["num_leaves", "min_data_in_leaf"]] = parametros[
    ["num_leaves", "min_data_in_leaf"]].astype(int)
tabla = validas.join(parametros, on="config")
print("\nValidación 2023 (8.760 horas), ordenado por MAE:")
columnas = ["config", "objective", "learning_rate", "num_leaves", "min_data_in_leaf",
            "MAE", "RMSE", "MAPE", "MASE", "sesgo", "MAE_rel_ingenuo", "arboles", "minutos"]
with pd.option_context("display.width", 200):
    print(tabla[columnas].round({"learning_rate": 4, "MAE": 1, "RMSE": 1, "MAPE": 2, "MASE": 3,
                                 "sesgo": 0, "MAE_rel_ingenuo": 3, "minutos": 1})
          .to_string(index=False))
print(f"Referencia: ingenuo estacional, MAE {mae_ingenuo:.1f} MW")
print(f"Configuraciones en el tope de árboles: {int(validas['tope'].sum())}")

# --- 2. Ganadora: previsiones en el formato común, importancias y cuantiles (11.2) ---
ganadora = validas.iloc[0]["config"]
print(f"\nGANADORA: {ganadora} {CONFIGURACIONES[ganadora]}")
tabla_g = pd.read_parquet(v.DIR_PREDICCIONES / f"lightgbm_{ganadora}_validacion.parquet")
tabla_g.assign(modelo="lightgbm").to_parquet(v.DIR_PREDICCIONES / "lightgbm_validacion.parquet",
                                             index=False)
modelo = modelos[ganadora] if ganadora in modelos else entrenar(
    X, y, fin, CONFIGURACIONES[ganadora])[0]
ganancia = pd.Series(modelo.feature_importance("gain"), index=modelo.feature_name())
print("Importancia por ganancia (% del total):")
print((100 * ganancia / ganancia.sum()).sort_values(ascending=False).round(1).to_string())

# Modelos de los cuantiles 0,05 y 0,95 con los hiperparámetros de la ganadora. Sus
# previsiones se guardan para la capa probabilística; la cobertura NO se calcula aquí.
cuantiles = {}
for alpha in (0.05, 0.95):
    inicio = time.perf_counter()
    m_q, arboles = entrenar(X, y, fin, CONFIGURACIONES[ganadora], alpha=alpha)
    cuantiles[alpha] = v.evaluar(hacer_prever(m_q), datos, "validacion", "x", guardar=False)
    print(f"Cuantil {alpha:.2f}".replace(".", ",") + f": {arboles} árboles ({(time.perf_counter() - inicio) / 60:.1f} min)")
inferior = cuantiles[0.05]["prevision"].to_numpy()
superior = cuantiles[0.95]["prevision"].to_numpy()
cruces = inferior > superior                                # regla de cruce (11.2.5)
intervalos = tabla_g[["hora_utc", "fecha_objetivo", "real"]].assign(
    prevision=tabla_g["prevision"], inferior=np.where(cruces, superior, inferior),
    superior=np.where(cruces, inferior, superior), modelo="lightgbm")
assert (cuantiles[0.05]["hora_utc"].to_numpy() == tabla_g["hora_utc"].to_numpy()).all()
intervalos.to_parquet(v.DIR_PREDICCIONES / "lightgbm_intervalos_validacion.parquet", index=False)
fuera = (intervalos["prevision"] < intervalos["inferior"]) | (
    intervalos["prevision"] > intervalos["superior"])
print(f"Horas con cuantiles cruzados (intercambiados): {int(cruces.sum())} de {len(cruces)}")
print(f"Horas con la previsión puntual fuera del intervalo: {int(fuera.sum())}")
