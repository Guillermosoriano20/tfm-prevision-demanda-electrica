"""Rejillas de 19 cuantiles de los cuatro modelos en 2023 (protocolo 14.3, 14.4 y 14.7).

Genera y guarda, para cada modelo, data/predicciones/{modelo}_cuantiles_validacion.parquet
(hora_utc, fecha_objetivo, real, prevision, q05 … q95), con las previsiones puntuales
oficiales en `prevision`. Antes de aceptar cada rejilla pasan los controles de 14.7.1 y la
prueba de fuga reforzada (demanda posterior al origen borrada) de todas las salidas nuevas;
si falla alguno, el script se detiene. Reanudable: un modelo ya guardado no se repite.

  ingenuo   cuantiles empíricos de los errores de 2017-2022 sin COVID, por bloque (14.3.1)
  sarimax   gaussianos con el error típico de get_forecast, parámetros FGLS fijos (14.3.2)
  lightgbm  19 modelos de cuantil con los hiperparámetros de c17 (14.4.3)
  nhits     segunda red con la pérdida multicuantil de 19 niveles y los de c13 (14.4.4)
"""
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, norm, uniform
from sklearn.model_selection import ParameterSampler

import lightgbm_modelo as lm
import nhits_modelo as nm
import nhits_rejilla as nr
import probabilistica as P
import validacion as v
from exogenas import exogenas_sarimax
from sarimax_fgls import ajustar_fgls, con_constante

MODELOS = ("ingenuo", "sarimax", "lightgbm", "nhits")
ORDEN_SARIMAX = (5, 0, 1)                    # ARMA(5,1) con `nivel` y retardos (10.10)
DIR_RESULTADOS = v.RAIZ / "data" / "resultados" / "probabilistica"
TOLERANCIA_SARIMAX = 1e-6                    # MW (14.7.1)


def cargar_ingenuo():
    """Módulo 15 (nombre con número: no se puede importar con `import`)."""
    ruta = Path(__file__).resolve().parent / "15_ingenuo_estacional.py"
    especificacion = importlib.util.spec_from_file_location("ingenuo_estacional", ruta)
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def tabla_oficial(modelo, datos):
    """Previsión puntual oficial en el formato común, con los controles de 14.7.1."""
    t = pd.read_parquet(v.DIR_PREDICCIONES / f"{modelo}_validacion.parquet")
    assert len(t) == 8760 and not t.isna().any().any(), f"{modelo}: filas o nulos"
    real = datos["demanda_mw"].reindex(pd.DatetimeIndex(t["hora_utc"]).as_unit(datos.index.unit))
    assert np.array_equal(real.to_numpy(), t["real"].to_numpy()), f"{modelo}: `real` distinta"
    return t[["hora_utc", "fecha_objetivo", "real", "prevision"]]


def con_cache(prever, datos):
    """Envoltorio: la primera llamada con los datos reales se guarda; así la tabla de salida
    sale de las mismas ejecuciones de la prueba de fuga, sin repetir el cálculo."""
    cache = {}

    def envuelta(d, origen):
        if d is not datos:
            return prever(d, origen)
        if origen.fecha_objetivo not in cache:
            cache[origen.fecha_objetivo] = prever(d, origen)
        return cache[origen.fecha_objetivo]
    return envuelta, cache


def recoger(cache, origenes, base):
    """Une las previsiones de los 365 orígenes, recortadas al día objetivo, en el orden de `base`."""
    trozos = [v.recortar(cache[o.fecha_objetivo], o) for o in origenes.itertuples()]
    t = pd.concat(trozos)
    assert np.array_equal(P.a_ns(t.index), P.a_ns(base["hora_utc"])), "horas distintas a las oficiales"
    return t


def sin_fuga(prever, datos, origenes, nombre):
    """Prueba de fuga reforzada (13): demanda posterior al origen borrada. 0 días o se detiene."""
    inicio = time.perf_counter()
    fugas = v.comprobar_fuga(prever, datos, origenes, modo="borrar")
    print(f"  prueba de fuga reforzada ({nombre}): {len(fugas)} días con fuga {fugas[:3]}"
          f" ({(time.perf_counter() - inicio) / 60:.1f} min)", flush=True)
    assert not fugas, f"FUGA DETECTADA en {nombre}: se detiene todo"


def guardar(modelo, base, Q, extra):
    """Reordena (14.4.5), comprueba y guarda la rejilla; devuelve la fila del resumen."""
    assert np.isfinite(Q).all(), f"{modelo}: valores no finitos"
    Q, reordenadas = P.reordenar(Q)
    t = base.copy()
    for j, c in enumerate(P.COLUMNAS_Q):
        t[c] = Q[:, j]
    t.to_parquet(v.DIR_PREDICCIONES / f"{modelo}_cuantiles_validacion.parquet", index=False)
    print(f"  {modelo}: rejilla guardada; horas reordenadas {reordenadas} de {len(t)}", flush=True)
    return {"modelo": modelo, "horas_reordenadas": reordenadas, **extra}


# ------------------------------------------------------------------------------ ingenuo ---
def rejilla_ingenuo(datos, origenes, base):
    modulo = cargar_ingenuo()
    t = v.evaluar(modulo.prever_ingenuo, datos, "validacion", "ingenuo", guardar=False)
    assert np.array_equal(t["prevision"].to_numpy(), base["prevision"].to_numpy()), \
        "el ingenuo recalculado no coincide con el guardado"
    mae = (base["real"] - base["prevision"]).abs().mean()
    assert abs(mae - 1400) <= 0.5, f"MAE del ingenuo {mae:.2f}, no 1.400"
    errores, bloques = P.errores_ingenuo(datos, modulo.prever_ingenuo, range(2017, 2023))
    Qb = P.cuantiles_por_bloque(errores, bloques)
    print(f"  ingenuo: MAE {mae:.1f} MW; {len(errores)} errores de 2017-2022 sin COVID"
          f" ({[int((bloques == b).sum()) for b in range(P.N_BLOQUES)]} por bloque)")
    print("  cuantiles del error por bloque (MW), q05 / q50 / q95:")
    for b in range(P.N_BLOQUES):
        print(f"    B{b + 1}: {Qb[b, 0]:8.0f} {Qb[b, 9]:8.0f} {Qb[b, 18]:8.0f}")
    _, bloque = P.anticipacion_y_bloque(base, origenes)
    Q = base["prevision"].to_numpy()[:, None] + Qb[bloque]
    return Q, {"MAE_mediana": mae, "n_errores": len(errores)}


# ------------------------------------------------------------------------------ SARIMAX ---
def preparar_sarimax(datos):
    """Demanda y exógenas del SARIMAX ganador, como en 18_sarimax_validacion.py (variante
    «nivel_retardos»). Devuelve funciones que dependen de los datos que se les pasen."""
    X = exogenas_sarimax(datos, k_dia=12, k_semana=12, k_anio=3, k_interaccion=3)
    X.index = X.index.tz_localize(None)
    X = X.asfreq("h")

    def preparar(d):
        y = d["demanda_mw"].where(~d["excluir_covid"])
        y.index = y.index.tz_localize(None)
        y = y.asfreq("h")
        nivel = y.shift(37).rolling(168, min_periods=168).mean()
        extra = pd.DataFrame({"nivel": nivel, "y_48": y.shift(48), "y_168": y.shift(168)})
        return y.where(extra["nivel"].notna()), pd.concat([X, extra.fillna(0.0)], axis=1)
    return preparar


def ajustar_sarimax(datos, fin):
    preparar = preparar_sarimax(datos)
    y, Xm = preparar(datos)
    corte = fin.tz_localize(None)
    res, informe = ajustar_fgls(y[y.index < corte], Xm[Xm.index < corte], order=ORDEN_SARIMAX)
    assert informe["convergio"], "el FGLS no convergió"
    return res, preparar, informe


def hacer_prever_sarimax(res, preparar):
    """prever(datos, origen): previsión y error típico de cada paso (get_forecast)."""
    def prever(d, origen):
        o, fin = origen.origen_utc.tz_localize(None), origen.fin_utc.tz_localize(None)
        y, Xm = preparar(d)
        Xm = con_constante(Xm)
        res_o = res.apply(y[y.index < o], exog=Xm[Xm.index < o])
        futuro = Xm[(Xm.index >= o) & (Xm.index < fin)]
        f = res_o.get_forecast(steps=len(futuro), exog=futuro)
        indice = futuro.index.tz_localize("UTC")
        return pd.DataFrame({"media": np.asarray(f.predicted_mean), "se": np.asarray(f.se_mean)},
                            index=indice)
    return prever


def rejilla_sarimax(datos, origenes, base, fin):
    inicio = time.perf_counter()
    res, preparar, informe = ajustar_sarimax(datos, fin)
    print(f"  sarimax: FGLS en {informe['vueltas']} vueltas, convergió"
          f" ({(time.perf_counter() - inicio) / 60:.1f} min)", flush=True)
    prever, cache = con_cache(hacer_prever_sarimax(res, preparar), datos)
    sin_fuga(prever, datos, origenes, "sarimax: media y error típico")
    t = recoger(cache, origenes, base)
    dif = np.abs(t["media"].to_numpy() - base["prevision"].to_numpy()).max()
    print(f"  sarimax: diferencia máxima con la previsión oficial {dif:.2e} MW")
    assert dif < TOLERANCIA_SARIMAX, "el SARIMAX reajustado no reproduce la previsión oficial"
    se = t["se"].to_numpy()
    assert (se > 0).all() and np.isfinite(se).all()
    Q = t["media"].to_numpy()[:, None] + norm.ppf(P.NIVELES)[None, :] * se[:, None]
    return Q, {"dif_max_oficial": dif, "se_medio": se.mean(), "minutos": (time.perf_counter() - inicio) / 60}


# ----------------------------------------------------------------------------- LightGBM ---
def parametros_c17():
    """Hiperparámetros de c17 (11.4), reconstruidos con el mismo sorteo y comprobados."""
    espacio = {"objective": ["regression", "regression_l1"], "learning_rate": loguniform(0.02, 0.2),
               "num_leaves": [15, 31, 63, 127, 255], "min_data_in_leaf": [20, 50, 100, 200, 500],
               "feature_fraction": uniform(0.5, 0.5), "bagging_fraction": uniform(0.5, 0.5),
               "lambda_l2": loguniform(1e-3, 10)}
    c = list(ParameterSampler(espacio, n_iter=19, random_state=42))[16]            # c17
    c = {k: (x if isinstance(x, str) else int(x) if k in ("num_leaves", "min_data_in_leaf")
             else float(x)) for k, x in c.items()}
    c["bagging_freq"] = 1
    assert (c["objective"], round(c["learning_rate"], 4), c["num_leaves"], c["min_data_in_leaf"],
            round(c["feature_fraction"], 4), round(c["bagging_fraction"], 4),
            round(c["lambda_l2"], 4)) == ("regression", 0.0318, 127, 20, 0.5554, 0.8553, 0.0572)
    return c


def hacer_prever_rejilla_lgb(modelos):
    """prever(datos, origen): las 19 predicciones de cuantil de las horas del día D+1."""
    def prever(d, origen):
        horas = pd.date_range(origen.inicio_utc, origen.fin_utc, freq="h",
                              inclusive="left").as_unit(d.index.unit)
        X = lm.variables(d, horas, pd.DatetimeIndex([origen.origen_utc] * len(horas)))
        assert not X.isna().any().any(), f"variables ausentes el {origen.fecha_objetivo}"
        return pd.DataFrame({c: m.predict(X) for c, m in zip(P.COLUMNAS_Q, modelos)}, index=horas)
    return prever


def rejilla_lightgbm(datos, origenes, base, fin):
    inicio = time.perf_counter()
    parametros = parametros_c17()
    X, y = lm.filas_entrenamiento(datos, fin)
    modelos, arboles = [], []
    for tau in P.NIVELES:
        t0 = time.perf_counter()
        m, n = lm.entrenar(X, y, fin, parametros, alpha=float(tau))
        modelos.append(m)
        arboles.append(n)
        print(f"    τ = {tau:.2f}: {n} árboles ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    prever, cache = con_cache(hacer_prever_rejilla_lgb(modelos), datos)
    sin_fuga(prever, datos, origenes, "lightgbm: 19 modelos de cuantil")
    t = recoger(cache, origenes, base)
    # Control 14.7.1: los modelos de 0,05 y 0,95 reproducen los guardados (11.4.6)
    guardados = pd.read_parquet(v.DIR_PREDICCIONES / "lightgbm_intervalos_validacion.parquet")
    assert np.array_equal(t["q05"].to_numpy(), guardados["inferior"].to_numpy()), "q05 distinto del guardado"
    assert np.array_equal(t["q95"].to_numpy(), guardados["superior"].to_numpy()), "q95 distinto del guardado"
    print("  lightgbm: los modelos de 0,05 y 0,95 reproducen los guardados (diferencia 0);"
          f" árboles {arboles[0]} y {arboles[-1]} (protocolo: 573 y 601)")
    return t[P.COLUMNAS_Q].to_numpy(), {"arboles": sum(arboles), "minutos": (time.perf_counter() - inicio) / 60}


# ------------------------------------------------------------------------------- N-HiTS ---
def rejilla_nhits(datos, origenes, base, fin):
    inicio = time.perf_counter()
    c = nr.config_c13()
    resto = {k: x for k, x in c.items() if k not in ("input_size", "ancho", "escalas")}
    nf, pasos, perdida = nr.entrenar(datos, fin, c["input_size"], semilla=nm.SEMILLA,
                                     **nm.arquitectura(c["ancho"], c["escalas"], **resto))
    print(f"    segunda red: {pasos} pasos, pérdida de control {perdida:.1f}"
          f" ({(time.perf_counter() - inicio) / 60:.1f} min)", flush=True)
    prever, cache = con_cache(nr.hacer_prever(nf), datos)
    sin_fuga(prever, datos, origenes, "nhits: red de 19 salidas")
    t = recoger(cache, origenes, base)
    y = base["real"].to_numpy()
    mae = np.abs(y - t["q50"].to_numpy()).mean()
    cobertura = ((t["q05"].to_numpy() <= y) & (y <= t["q95"].to_numpy())).mean()
    oficial = pd.read_parquet(v.DIR_PREDICCIONES / "nhits_intervalos_validacion.parquet")
    cobertura_of = ((oficial["inferior"] <= oficial["real"]) & (oficial["real"] <= oficial["superior"])).mean()
    print(f"  nhits: MAE de la mediana de la segunda red {mae:.1f} MW (oficial: "
          f"{(base['real'] - base['prevision']).abs().mean():.1f} MW); cobertura 0,05-0,95: "
          f"{100 * cobertura:.1f} % (red oficial: {100 * cobertura_of:.1f} %)")
    return t[P.COLUMNAS_Q].to_numpy(), {"pasos": pasos, "MAE_mediana": mae, "cobertura_90": cobertura,
                                        "cobertura_90_oficial": cobertura_of,
                                        "minutos": (time.perf_counter() - inicio) / 60}


# -------------------------------------------------------------------------------- main ---
def main():
    datos = v.cargar_datos()                                  # huella comprobada; 2024 cortado
    origenes = v.generar_origenes("validacion")
    otros = P.origenes_anio(2023)
    assert all(np.array_equal(origenes[c].to_numpy(), otros[c].to_numpy()) for c in origenes.columns), \
        "origenes_anio(2023) no coincide con generar_origenes"
    fin = origenes["origen_utc"].iloc[0]
    v.DIR_PREDICCIONES.mkdir(parents=True, exist_ok=True)
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    bases = {m: tabla_oficial(m, datos) for m in MODELOS}
    for m in MODELOS[1:]:                                      # mismas horas y misma demanda en los cuatro
        assert np.array_equal(P.a_ns(bases[m]["hora_utc"]), P.a_ns(bases["ingenuo"]["hora_utc"]))
    print("Controles de entrada: 4 archivos oficiales de 8.760 filas, sin nulos y con `real` idéntica"
          " al dataset", flush=True)

    constructores = {"ingenuo": lambda b: rejilla_ingenuo(datos, origenes, b),
                     "sarimax": lambda b: rejilla_sarimax(datos, origenes, b, fin),
                     "lightgbm": lambda b: rejilla_lightgbm(datos, origenes, b, fin),
                     "nhits": lambda b: rejilla_nhits(datos, origenes, b, fin)}
    resumen = []
    for modelo in MODELOS:
        archivo = v.DIR_PREDICCIONES / f"{modelo}_cuantiles_validacion.parquet"
        if archivo.exists():
            print(f"{modelo}: ya guardada ({archivo.name}); no se repite", flush=True)
            continue
        print(f"{modelo}:", flush=True)
        Q, extra = constructores[modelo](bases[modelo])
        resumen.append(guardar(modelo, bases[modelo], Q, extra))
    if resumen:
        nuevo = pd.DataFrame(resumen)
        ruta = DIR_RESULTADOS / "rejillas_resumen.csv"
        if ruta.exists():
            nuevo = pd.concat([pd.read_csv(ruta), nuevo]).drop_duplicates("modelo", keep="last")
        nuevo.to_csv(ruta, index=False)
        print("\nResumen:\n" + nuevo.round(3).to_string(index=False))
    print("\nRejillas listas: ejecutar 27_capa_probabilistica.py")


if __name__ == "__main__":
    main()
