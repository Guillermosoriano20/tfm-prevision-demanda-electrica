"""LightGBM: variables, filas de entrenamiento, ajuste y previsión.

Implementa docs/protocolo_oe5.md, apartados 11.1 (variables, commit caa6f3e) y 11.2
(intervalos por regresión cuantílica, commit e670cc2). Módulo sin número porque se
importa, como validacion.py y exogenas.py.

Modelo directo: una fila por hora objetivo t, con su origen (12:00 locales del día
anterior al día local de t). La misma función `variables` construye las filas de
entrenamiento y las de previsión, para que no puedan divergir.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

import validacion as v
from exogenas import grados_dia, navidad

RETARDOS = (48, 72, 96, 120, 144, 168, 336)       # horas UTC, todos >= 37 (11.1.2)
VENTANA_COVID = 336                                # todas las entradas en [t-336, t-12]
COLUMNAS = ([f"y_{L}" for L in RETARDOS] + ["nivel", "y_ultima", "y_media24", "anticipacion",
            "gd_calefaccion", "gd_refrigeracion", "t_ponderada", "hora_del_dia",
            "dia_semana", "dia_anio", "fraccion_festivo", "puente", "navidad"])
FIJOS = {"seed": 42, "deterministic": True, "force_row_wise": True, "num_threads": 4,
         "verbose": -1}                            # protocolo 3.5: semilla 42, determinista
HORAS_CONTROL = 8 * 168                             # últimas 8 semanas (protocolo 2.5, 11.3)
MAX_ARBOLES, PACIENCIA = 10_000, 100               # parada temprana (11.3)


def _en(serie, instantes):
    """Valores de una serie horaria (índice UTC) en los instantes pedidos; NaN si no está."""
    return serie.reindex(pd.DatetimeIndex(instantes).as_unit(serie.index.unit)).to_numpy()


def origen_de(horas, datos):
    """Origen de cada hora objetivo: 12:00 locales del día anterior a su fecha local."""
    fecha = datos.loc[horas, "fecha_local"]
    local = pd.DatetimeIndex(fecha - v.UN_DIA + v.HORA_ORIGEN * v.UNA_HORA)
    return local.tz_localize(v.ZONA).tz_convert("UTC")


def variables(datos, horas, origenes):
    """Las 20 variables del apartado 11.1 para las horas objetivo dadas.

    `horas` y `origenes`: índices UTC de la misma longitud. Todo lo que procede de la
    demanda se calcula aquí, a partir de `datos`, para que la prueba de fuga sea efectiva
    (11.1.8). Ventanas móviles siempre hacia atrás.
    """
    horas, origenes = pd.DatetimeIndex(horas), pd.DatetimeIndex(origenes)
    y = datos["demanda_mw"]
    X = {f"y_{L}": _en(y, horas - L * v.UNA_HORA) for L in RETARDOS}
    X["nivel"] = _en(y.rolling(168, min_periods=168).mean(), horas - 37 * v.UNA_HORA)
    X["y_ultima"] = _en(y, origenes - v.UNA_HORA)                     # 11:00-12:00 de D
    X["y_media24"] = _en(y.rolling(24, min_periods=24).mean(), origenes - v.UNA_HORA)
    X["anticipacion"] = ((horas - origenes) / v.UNA_HORA).to_numpy()
    filas = datos.loc[horas]
    gd = grados_dia(datos).loc[horas]           # media del día local completo (todas sus horas)
    X["gd_calefaccion"] = gd["gd_calefaccion"].to_numpy()
    X["gd_refrigeracion"] = gd["gd_refrigeracion"].to_numpy()
    X["t_ponderada"] = filas["t_ponderada"].to_numpy()
    X["hora_del_dia"] = filas["hora_del_dia"].to_numpy()
    X["dia_semana"] = filas["dia_semana"].to_numpy()
    X["dia_anio"] = filas["fecha_local"].dt.dayofyear.to_numpy()
    X["fraccion_festivo"] = filas["fraccion_festivo"].to_numpy()
    X["puente"] = filas["puente"].astype(float).to_numpy()
    X["navidad"] = navidad(filas).to_numpy()
    return pd.DataFrame(X, index=horas)[COLUMNAS].astype(float)


def filas_entrenamiento(datos, fin):
    """Filas (X, y) con hora objetivo terminada en `fin` (11.1.7) y COVID estricto (11.1.6)."""
    horas = datos.index[datos.index + v.UNA_HORA <= fin]
    covid = datos["excluir_covid"].astype(float)
    toca_covid = covid.rolling(VENTANA_COVID + 1, min_periods=1).max()   # [t-336, t]
    horas = horas[_en(toca_covid, horas) == 0]
    X = variables(datos, horas, origen_de(horas, datos))
    X = X.dropna()                                  # entradas ausentes (inicio de 2017): fuera
    y = datos.loc[X.index, "demanda_mw"]
    assert (X.index + v.UNA_HORA <= fin).all()
    return X, y


def ajustar(X, y, parametros, alpha=None, num_arboles=100):
    """Ajusta un modelo: puntual (alpha=None) o del cuantil alpha (11.2)."""
    p = {**parametros, **FIJOS}
    if alpha is not None:
        p.update(objective="quantile", alpha=alpha)
    return lgb.train(p, lgb.Dataset(X, label=y), num_boost_round=num_arboles)


def entrenar(X, y, fin, parametros, alpha=None):
    """Procedimiento del apartado 11.3: parada temprana y reajuste con toda la ventana.

    1. Árboles por parada temprana: entrena con las filas anteriores a las últimas
       8 semanas y para tras 100 árboles sin mejora en ellas (MAE en el modelo puntual,
       pinball en los cuantiles); máximo 10.000. 2023 no interviene.
    2. Reajuste con todas las filas (incluidas esas 8 semanas) y ese número de árboles.
    Devuelve el modelo reajustado y el número de árboles.
    """
    control = X.index >= fin - HORAS_CONTROL * v.UNA_HORA
    p = {**parametros, **FIJOS, "metric": "l1" if alpha is None else "quantile"}
    if alpha is not None:
        p.update(objective="quantile", alpha=alpha)
    previo = lgb.train(p, lgb.Dataset(X[~control], label=y[~control]), MAX_ARBOLES,
                       valid_sets=[lgb.Dataset(X[control], label=y[control])],
                       callbacks=[lgb.early_stopping(PACIENCIA, verbose=False)])
    arboles = previo.best_iteration
    return ajustar(X, y, parametros, alpha=alpha, num_arboles=arboles), arboles


def hacer_prever(modelo):
    """Interfaz común prever(datos, origen): previsión de las horas del día local D+1."""
    def prever(datos, origen):
        horas = pd.date_range(origen.inicio_utc, origen.fin_utc, freq="h", inclusive="left")
        horas = horas.as_unit(datos.index.unit)
        origenes = pd.DatetimeIndex([origen.origen_utc] * len(horas))
        X = variables(datos, horas, origenes)
        assert not X.isna().any().any(), f"variables ausentes el {origen.fecha_objetivo}"
        return pd.Series(modelo.predict(X), index=horas)
    return prever
