"""N-HiTS: búsqueda aleatoria de 20 configuraciones evaluadas en 2023 (validación).

Protocolo: docs/protocolo_oe5.md, apartado 12.5 (espacio y procedimiento registrados con
commit antes de lanzar); diseño 12.1, intervalos 12.2, máximos 12.4.7. Cada
configuración: parada temprana con las últimas 8 semanas de 2017-2022, reentrenamiento
con toda la ventana y previsión de los 365 orígenes de 2023 sin reajustes. Decide el
menor MAE de la mediana. La ganadora se reentrena con las semillas 43 y 44 (ruido,
protocolo 3.5) y pasa la prueba de fuga reforzada. Cada configuración se guarda al
terminar: si se interrumpe, al relanzarlo no se repiten las ya hechas.
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.model_selection import ParameterSampler

import validacion as v
import nhits_modelo as nm

# --- Espacio de búsqueda (12.5.2): 19 sorteadas con semilla 42 + la referencia ---
ESPACIO = {
    "input_size": [168, 336, 504],
    "learning_rate": loguniform(1e-4, 3e-3),
    "ancho": [128, 256, 512],
    "windows_batch_size": [256, 512, 1024],
    "escalas": ["A", "B", "C"],
    "dropout_prob_theta": [0.0, 0.1],
}
CONFIGURACIONES = {"c00": {"input_size": 336, "learning_rate": 1e-3, "ancho": 512,
                           "windows_batch_size": 1024, "escalas": "A",
                           "dropout_prob_theta": 0.0}}     # c00: valores por defecto
for i, c in enumerate(ParameterSampler(ESPACIO, n_iter=19, random_state=42), 1):
    CONFIGURACIONES[f"c{i:02d}"] = {k: (x if isinstance(x, str) else float(x)
                                        if k in ("learning_rate", "dropout_prob_theta")
                                        else int(x)) for k, x in c.items()}

# Comprobación: el sorteo reproduce el registrado en el protocolo (12.5.3)
TASAS_REGISTRADAS = [0.00121, 0.00014, 0.00012, 0.00028, 0.0006, 0.00014, 0.00012, 0.0001,
                     0.00011, 0.0017, 0.00042, 0.00271, 0.00119, 0.0003, 0.00267, 0.00013,
                     0.0016, 0.00013, 0.00083]
tasas = [round(CONFIGURACIONES[f"c{i:02d}"]["learning_rate"], 5) for i in range(1, 20)]
assert tasas == TASAS_REGISTRADAS, "el sorteo no coincide con el registrado"
SEMILLAS_RUIDO = (43, 44)                                  # la 42 es la de la búsqueda


def entrenar_config(c, semilla=nm.SEMILLA):
    resto = {k: x for k, x in c.items() if k not in ("input_size", "ancho", "escalas")}
    return nm.entrenar(datos, fin, c["input_size"], semilla=semilla,
                       **nm.arquitectura(c["ancho"], c["escalas"], **resto))


def prever_2023(nf, modelo):
    """Las tres salidas de los 365 orígenes recortadas al día D+1, en el formato común."""
    prever, trozos = nm.hacer_prever(nf), []
    for origen in origenes.itertuples():
        p = v.recortar(prever(datos, origen), origen)
        trozos.append(pd.DataFrame({
            "hora_utc": p.index, "fecha_objetivo": origen.fecha_objetivo,
            "real": datos.loc[p.index, "demanda_mw"].to_numpy(),
            "prevision": p["mediana"].to_numpy(),
            "q05": p["q05"].to_numpy(), "q95": p["q95"].to_numpy()}))
    return pd.concat(trozos, ignore_index=True).assign(modelo=modelo)


def resumen_de(tabla):
    m = v.metricas(tabla, escala)
    return {**m.to_dict(), "sesgo": (tabla["real"] - tabla["prevision"]).mean(),
            "MAE_rel_ingenuo": m["MAE"] / mae_ingenuo,
            "cruces": int((tabla["q05"] > tabla["q95"]).sum())}


datos = v.cargar_datos()                                   # huella comprobada; 2024 cortado
origenes = v.generar_origenes("validacion")
fin = origenes["origen_utc"].iloc[0]                       # 31/12/2022 12:00 local
escala = v.escala_mase(datos)
ingenuo = pd.read_parquet(v.DIR_PREDICCIONES / "ingenuo_validacion.parquet")
mae_ingenuo = (ingenuo["real"] - ingenuo["prevision"]).abs().mean()
RESUMEN = v.RAIZ / "data" / "interim" / "nhits_busqueda.parquet"
hechas = pd.read_parquet(RESUMEN).set_index("config") if RESUMEN.exists() else None
v.DIR_PREDICCIONES.mkdir(parents=True, exist_ok=True)
RESUMEN.parent.mkdir(parents=True, exist_ok=True)

# --- 1. Las 20 configuraciones ---
filas, modelos = [], {}
for nombre, c in CONFIGURACIONES.items():
    archivo = v.DIR_PREDICCIONES / f"nhits_{nombre}_validacion.parquet"
    if archivo.exists() and hechas is not None and nombre in hechas.index:
        anterior = hechas.loc[nombre]
        filas.append({"config": nombre, **anterior.to_dict()})
        print(f"  {nombre}: guardada, MAE {anterior['MAE']:.1f} MW", flush=True)
        continue
    inicio = time.perf_counter()
    try:
        nf, pasos, perdida = entrenar_config(c)
        tabla = prever_2023(nf, f"nhits_{nombre}")
    except Exception as error:                     # se registra como fallida (12.5)
        print(f"  {nombre}: FALLIDA ({error})", flush=True)
        filas.append({"config": nombre, "estado": "fallida"})
        pd.DataFrame(filas).to_parquet(RESUMEN, index=False)
        continue
    tabla.to_parquet(archivo, index=False)
    filas.append({"config": nombre, "estado": "valida", **resumen_de(tabla), "pasos": pasos,
                  "perdida_control": perdida, "tope": pasos == nm.P_MAX,
                  "minutos": (time.perf_counter() - inicio) / 60})
    modelos[nombre] = nf
    pd.DataFrame(filas).to_parquet(RESUMEN, index=False)
    print(f"  {nombre}: MAE {filas[-1]['MAE']:.1f} MW, sesgo {filas[-1]['sesgo']:.0f} MW,"
          f" {pasos} pasos ({filas[-1]['minutos']:.1f} min)", flush=True)

resumen = pd.DataFrame(filas)
resumen.to_parquet(RESUMEN, index=False)
validas = resumen[resumen["estado"] == "valida"].sort_values("MAE")
tabla = validas.join(pd.DataFrame.from_dict(CONFIGURACIONES, orient="index"), on="config")
print("\nValidación 2023 (8.760 horas), ordenado por MAE de la mediana:")
columnas = ["config", "input_size", "learning_rate", "ancho", "windows_batch_size",
            "escalas", "dropout_prob_theta", "MAE", "RMSE", "MAPE", "MASE", "sesgo",
            "MAE_rel_ingenuo", "pasos", "cruces", "minutos"]
with pd.option_context("display.width", 250):
    print(tabla[columnas].round({"learning_rate": 5, "MAE": 1, "RMSE": 1, "MAPE": 2,
                                 "MASE": 3, "sesgo": 0, "MAE_rel_ingenuo": 3, "minutos": 1})
          .to_string(index=False))
print(f"Referencia: ingenuo estacional, MAE {mae_ingenuo:.1f} MW")
print(f"Configuraciones en el tope de {nm.P_MAX} pasos: {int(validas['tope'].sum())};"
      f" fallidas: {int((resumen['estado'] == 'fallida').sum())}")

# --- 2. Ganadora: formato común, intervalos (12.2.5) y prueba de fuga reforzada ---
ganadora = validas.iloc[0]["config"]
print(f"\nGANADORA: {ganadora} {CONFIGURACIONES[ganadora]}")
tabla_g = pd.read_parquet(v.DIR_PREDICCIONES / f"nhits_{ganadora}_validacion.parquet")
tabla_g[["hora_utc", "fecha_objetivo", "real", "prevision"]].assign(modelo="nhits").to_parquet(
    v.DIR_PREDICCIONES / "nhits_validacion.parquet", index=False)
cruces = tabla_g["q05"] > tabla_g["q95"]                   # regla de cruce (12.2.5)
intervalos = tabla_g[["hora_utc", "fecha_objetivo", "real", "prevision"]].assign(
    inferior=np.where(cruces, tabla_g["q95"], tabla_g["q05"]),
    superior=np.where(cruces, tabla_g["q05"], tabla_g["q95"]), modelo="nhits")
intervalos.to_parquet(v.DIR_PREDICCIONES / "nhits_intervalos_validacion.parquet", index=False)
fuera = (intervalos["prevision"] < intervalos["inferior"]) | (
    intervalos["prevision"] > intervalos["superior"])
print(f"Horas con cuantiles cruzados (intercambiados): {int(cruces.sum())} de {len(cruces)}")
print(f"Horas con la mediana fuera del intervalo: {int(fuera.sum())}")

nf_g = modelos[ganadora] if ganadora in modelos else entrenar_config(
    CONFIGURACIONES[ganadora])[0]
inicio = time.perf_counter()
fugas = v.comprobar_fuga(nm.hacer_prever(nf_g), datos, origenes, modo="borrar")
print(f"Prueba de fuga reforzada con la ganadora: {len(fugas)} días con fuga {fugas[:3]}"
      f" ({time.perf_counter() - inicio:.0f} s)")
assert not fugas, "FUGA DETECTADA: se detiene todo"

# --- 3. Ruido entre semillas (protocolo 3.5): la previsión oficial es la de la 42 ---
maes = {nm.SEMILLA: float(validas.iloc[0]["MAE"])}
for semilla in SEMILLAS_RUIDO:
    archivo = v.DIR_PREDICCIONES / f"nhits_{ganadora}_s{semilla}_validacion.parquet"
    if archivo.exists():
        tabla_s = pd.read_parquet(archivo)
    else:
        inicio = time.perf_counter()
        nf_s, pasos, _ = entrenar_config(CONFIGURACIONES[ganadora], semilla=semilla)
        tabla_s = prever_2023(nf_s, f"nhits_{ganadora}_s{semilla}")
        tabla_s.to_parquet(archivo, index=False)
        print(f"  semilla {semilla}: {pasos} pasos"
              f" ({(time.perf_counter() - inicio) / 60:.1f} min)", flush=True)
    maes[semilla] = float(resumen_de(tabla_s)["MAE"])
valores = np.array(list(maes.values()))
print("MAE de la ganadora por semilla: "
      + ", ".join(f"{s}: {m:.1f} MW" for s, m in maes.items()))
print(f"  rango {valores.max() - valores.min():.1f} MW, desviación típica"
      f" {valores.std(ddof=1):.1f} MW")
if len(validas) > 1:
    print(f"  distancia entre la ganadora y la segunda: "
          f"{validas.iloc[1]['MAE'] - validas.iloc[0]['MAE']:.1f} MW")
