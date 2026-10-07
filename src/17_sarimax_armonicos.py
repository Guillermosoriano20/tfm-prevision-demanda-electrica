"""SARIMAX: armónicos de Fourier e interacciones por AIC con 2017-2022.

Criterio fijado antes de medir: docs/protocolo_oe5.md, apartado 9 (commit 54d8229).
Solo usa la ventana de ajuste (hasta el 31/12/2022 12:00); 2023 no interviene.
Cada candidato se guarda en disco al terminar: si se interrumpe, al relanzarlo
continúa donde se quedó.
"""
import itertools
import time

import numpy as np
import pandas as pd

import validacion as v
from exogenas import exogenas_sarimax
from sarimax_fgls import ajustar_fgls

RESULTADOS = v.RAIZ / "data" / "interim" / "sarimax_armonicos.parquet"
REJILLA = {"dia": [4, 6, 8, 10], "semana": [4, 6, 8, 10], "anio": [2, 3, 4], "inter": [0, 3]}
AMPLIACION = {"dia": [12], "semana": [12], "anio": [5, 6]}   # regla de borde (9.6)

datos = v.cargar_datos()
y = datos["demanda_mw"].where(~datos["excluir_covid"])
y.index = y.index.tz_localize(None)
y = y.asfreq("h")
fin_ajuste = v.generar_origenes("validacion")["origen_utc"].iloc[0].tz_localize(None)
y = y[y.index < fin_ajuste]
n_obs = int(y.notna().sum())
print(f"Ventana: {len(y)} horas, {n_obs} con dato (sin COVID)")

tabla = pd.read_parquet(RESULTADOS) if RESULTADOS.exists() else pd.DataFrame()


def evaluar(dia, semana, anio, inter):
    """Ajusta un candidato por FGLS y devuelve su fila (o la ya guardada)."""
    global tabla
    clave = {"dia": dia, "semana": semana, "anio": anio, "inter": inter}
    if len(tabla):
        hecho = tabla[(tabla[list(clave)] == pd.Series(clave)).all(axis=1)]
        if len(hecho):
            return hecho.iloc[0]
    X = exogenas_sarimax(datos, dia, semana, anio, inter)
    X.index = X.index.tz_localize(None)
    X = X.asfreq("h")
    X = X[X.index < fin_ajuste]
    inicio = time.perf_counter()
    _, informe = ajustar_fgls(y, X, order=(2, 0, 1))
    k = X.shape[1] + 1 + 3 + 1          # exógenas + constante + ARMA(2,1) + varianza
    fila = pd.Series({**clave, "columnas": X.shape[1], "parametros": k,
                      "llf": informe["llf"], "AIC": -2 * informe["llf"] + 2 * k,
                      "BIC": -2 * informe["llf"] + k * np.log(n_obs),
                      "vueltas": informe["vueltas"], "convergio": informe["convergio"],
                      "minutos": (time.perf_counter() - inicio) / 60})
    tabla = pd.concat([tabla, pd.DataFrame([fila.to_dict()])], ignore_index=True)
    tabla.to_parquet(RESULTADOS, index=False)
    print(f"  dia={dia:2} semana={semana:2} anio={anio} inter={inter}: AIC {fila['AIC']:.1f}"
          f" ({fila['minutos']:.1f} min, {informe['vueltas']} vueltas,"
          f" convergió {informe['convergio']})", flush=True)
    return fila


# --- Rejilla completa (9.4) ---
combinaciones = list(itertools.product(*REJILLA.values()))
print(f"Rejilla: {len(combinaciones)} candidatos")
filas = [evaluar(*c) for c in combinaciones]
ganador = pd.DataFrame(filas).sort_values("AIC").iloc[0]

# --- Regla de borde (9.6): una sola ampliación por dimensión, con las demás fijas ---
for dim, extra in AMPLIACION.items():
    if ganador[dim] != max(REJILLA[dim]):
        continue
    print(f"Borde en '{dim}' ({ganador[dim]}): se amplía a {extra}")
    clave = {d: int(ganador[d]) for d in REJILLA}
    nuevas = [evaluar(**{**clave, dim: valor}) for valor in extra]
    ganador = pd.DataFrame([ganador, *nuevas]).sort_values("AIC").iloc[0]
    if ganador[dim] == max(extra):
        print(f"  vuelve a caer en el borde ({ganador[dim]}): se acepta y se declara")

# --- Resumen ---
todos = tabla.sort_values("AIC").reset_index(drop=True)
todos["dif_AIC"] = todos["AIC"] - todos["AIC"].iloc[0]
todos["dif_BIC"] = todos["BIC"] - todos["BIC"].min()
print("\nLos 10 mejores por AIC:")
print(todos.head(10)[["dia", "semana", "anio", "inter", "columnas", "dif_AIC", "dif_BIC",
                      "minutos"]].round(1).to_string(index=False))
print(f"\nGANADOR: dia={int(ganador['dia'])}, semana={int(ganador['semana'])},"
      f" anio={int(ganador['anio'])}, inter={int(ganador['inter'])}"
      f" ({int(ganador['columnas'])} columnas exógenas)")
print(f"Mejor por BIC (solo informativo): {todos.sort_values('BIC').iloc[0][['dia', 'semana', 'anio', 'inter']].astype(int).to_dict()}")
print(f"Tiempo total de los ajustes: {tabla['minutos'].sum():.0f} min")
