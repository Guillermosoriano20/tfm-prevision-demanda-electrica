"""SARIMAX: prueba de fuga y evaluación de la rejilla en 2023 (validación).

Rejilla: docs/protocolo_oe5.md, apartado 10 (commit f398409) con las enmiendas 10.8
(exógena `nivel`, commit 4605891) y 10.9 (retardos de 48 y 168 h): 15 configuraciones
ARMA(p, q) con `nivel` y retardos, más las dos referencias ARMA(2,1) evaluadas antes
(sin `nivel` y con `nivel` sin retardos), que cuentan en el presupuesto (17 de 20).
Cada configuración se ajusta una vez con datos hasta el 31/12/2022 12:00 y prevé los
365 orígenes de 2023 con parámetros fijos. Decide el menor MAE. Las previsiones de cada
configuración se guardan en data/predicciones/: si se interrumpe, al relanzarlo no se
repiten las ya hechas.
"""
import time

import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

import validacion as v
from exogenas import exogenas_sarimax
from sarimax_fgls import ajustar_fgls, con_constante

# (p, q, variante): "sin_nivel", "nivel" o "nivel_retardos"
CONFIGURACIONES = ([(2, 1, "sin_nivel"), (2, 1, "nivel")]
                   + [(p, q, "nivel_retardos") for p in (1, 2, 3, 4, 5) for q in (0, 1, 2)])
REFERENCIA_FUGA = (2, 1, "nivel_retardos")

datos = v.cargar_datos()                                   # huella comprobada; 2024 cortado
X = exogenas_sarimax(datos, k_dia=12, k_semana=12, k_anio=3, k_interaccion=3)
X.index = X.index.tz_localize(None)                        # statsmodels: índice sin zona
X = X.asfreq("h")
origenes = v.generar_origenes("validacion")
fin_ajuste = origenes["origen_utc"].iloc[0].tz_localize(None)   # 31/12/2022 12:00 local


def demanda(d):
    """Demanda con COVID vacío, índice horario sin zona."""
    y = d["demanda_mw"].where(~d["excluir_covid"])
    y.index = y.index.tz_localize(None)
    return y.asfreq("h")


def variable_nivel(y):
    """Media de la demanda de t-204 a t-37 h (apartado 10.8). Vacía si falta alguna hora."""
    return y.shift(37).rolling(168, min_periods=168).mean()


def preparar(d, variante):
    """Demanda y exógenas (sin constante) de un modelo; horas sin `nivel` excluidas."""
    y = demanda(d)
    if variante == "sin_nivel":
        return y, X
    extra = {"nivel": variable_nivel(y)}
    if variante == "nivel_retardos":                       # enmienda 10.9: retardos >= 37 h
        extra["y_48"], extra["y_168"] = y.shift(48), y.shift(168)
    extra = pd.DataFrame(extra)
    # Donde `nivel` falta (su ventana contiene t-48 y t-168), la hora no entra en el
    # ajuste; el valor de relleno no se usa
    return y.where(extra["nivel"].notna()), pd.concat([X, extra.fillna(0.0)], axis=1)


def ajustar(p, q, variante):
    y, Xm = preparar(datos, variante)
    return ajustar_fgls(y[y.index < fin_ajuste], Xm[Xm.index < fin_ajuste], order=(p, 0, q))


def hacer_prever(res, variante):
    """Interfaz común prever(datos, origen): solo usa la demanda de horas ya terminadas."""
    def prever(d, origen):
        o = origen.origen_utc.tz_localize(None)
        fin = origen.fin_utc.tz_localize(None)                       # fin del día local D+1
        y, Xm = preparar(d, variante)
        Xm = con_constante(Xm)
        res_o = res.apply(y[y.index < o], exog=Xm[Xm.index < o])
        futuro = Xm[(Xm.index >= o) & (Xm.index < fin)]             # 35, 36 o 37 pasos
        prevision = res_o.forecast(steps=len(futuro), exog=futuro)
        prevision.index = prevision.index.tz_localize("UTC")
        return prevision
    return prever


def nombre(p, q, variante):
    return f"sarimax_p{p}q{q}_{variante}"


# --- 1. Prueba de fuga con ARMA(2,1) con `nivel` y retardos (apartado 10.9) ---
inicio = time.perf_counter()
res_ref, _ = ajustar(*REFERENCIA_FUGA)
fugas = v.comprobar_fuga(hacer_prever(res_ref, REFERENCIA_FUGA[2]), datos, origenes)
print(f"Prueba de fuga ARMA(2,1) con nivel y retardos: {len(fugas)} días con fuga {fugas[:3]}"
      f" ({(time.perf_counter() - inicio) / 60:.1f} min)", flush=True)
assert not fugas, "FUGA DETECTADA: se detiene la evaluación"

# --- 2. Evaluación de las 17 configuraciones en 2023 ---
escala = v.escala_mase(datos)
ingenuo = pd.read_parquet(v.DIR_PREDICCIONES / "ingenuo_validacion.parquet")
mae_ingenuo = (ingenuo["real"] - ingenuo["prevision"]).abs().mean()
filas, ajustes = [], {REFERENCIA_FUGA: res_ref}
for p, q, variante in CONFIGURACIONES:
    etiqueta = f"ARMA({p},{q}) {variante}"
    archivo = v.DIR_PREDICCIONES / f"{nombre(p, q, variante)}_validacion.parquet"
    inicio = time.perf_counter()
    try:
        if archivo.exists():
            tabla, estado = pd.read_parquet(archivo), "guardada"
        else:
            if (p, q, variante) in ajustes:
                res = ajustes[(p, q, variante)]
            else:
                res, informe = ajustar(p, q, variante)
                if not informe["convergio"]:
                    raise RuntimeError("el FGLS no convergió en 20 vueltas")
                ajustes[(p, q, variante)] = res
            tabla = v.evaluar(hacer_prever(res, variante), datos, "validacion",
                              nombre(p, q, variante))
            estado = "nueva"
    except Exception as error:                     # se registra como fallida (10.6)
        print(f"  {etiqueta}: FALLIDA ({error})", flush=True)
        filas.append({"p": p, "q": q, "variante": variante, "estado": "fallida"})
        continue
    m = v.metricas(tabla, escala)
    sesgo = (tabla["real"] - tabla["prevision"]).mean()
    filas.append({"p": p, "q": q, "variante": variante, "estado": estado, **m.to_dict(),
                  "sesgo": sesgo, "MAE_rel_ingenuo": m["MAE"] / mae_ingenuo,
                  "minutos": (time.perf_counter() - inicio) / 60})
    print(f"  {etiqueta}: MAE {m['MAE']:.1f} MW, sesgo {sesgo:.0f} MW"
          f" ({filas[-1]['minutos']:.1f} min, {estado})", flush=True)

resumen = pd.DataFrame(filas)
validas = resumen[resumen["estado"] != "fallida"].sort_values("MAE")
print("\nValidación 2023 (8.760 horas), ordenado por MAE:")
print(validas[["p", "q", "variante", "MAE", "RMSE", "MAPE", "MASE", "sesgo", "MAE_rel_ingenuo"]]
      .round({"MAE": 1, "RMSE": 1, "MAPE": 2, "MASE": 3, "sesgo": 0, "MAE_rel_ingenuo": 3})
      .to_string(index=False))
print(f"Referencia: ingenuo estacional, MAE {mae_ingenuo:.1f} MW")

# --- 3. Ganadora: previsiones en el formato común y diagnóstico (apartado 10.7) ---
mejor = validas.iloc[0]
p, q, variante = int(mejor["p"]), int(mejor["q"]), mejor["variante"]
print(f"\nGANADORA: ARMA({p},{q}) {variante}")
ganadora = pd.read_parquet(v.DIR_PREDICCIONES / f"{nombre(p, q, variante)}_validacion.parquet")
ganadora.assign(modelo="sarimax").to_parquet(v.DIR_PREDICCIONES / "sarimax_validacion.parquet",
                                              index=False)
res = ajustes[(p, q, variante)] if (p, q, variante) in ajustes else ajustar(p, q, variante)[0]
r = pd.Series(res.filter_results.standardized_forecasts_error[0]).dropna()
r = r.iloc[p + q + 1:]                                     # se descartan las primeras horas
print("Ljung-Box sobre los residuos estandarizados del ajuste 2017-2022:")
print(acorr_ljungbox(r, lags=[24, 168]).round(4).to_string())
adf = adfuller(r, maxlag=48, autolag="AIC", result_object=False)
kp = kpss(r, regression="c", nlags="auto", result_object=False)
print(f"ADF: estadístico {adf[0]:.2f}, p-valor {adf[1]:.4f}; KPSS: estadístico {kp[0]:.3f},"
      f" p-valor {kp[1]:.3f}")
print(f"Autocorrelación de los residuos: k=1 {r.autocorr(1):.3f}, k=24 {r.autocorr(24):.3f},"
      f" k=168 {r.autocorr(168):.3f}")
