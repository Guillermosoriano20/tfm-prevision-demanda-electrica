"""Estimación del SARIMAX (regresión con errores ARMA) por FGLS iterado.

Método fijado en docs/protocolo_oe5.md, apartado 8.6 (commit 8ec9995): el ajuste
conjunto por máxima verosimilitud no converge con unas 60 exógenas y 52.000 horas.
Módulo sin número porque se importa, como validacion.py.
"""
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


def con_constante(X):
    """Añade la columna de la constante; el modelo completo la usa como una exógena más."""
    return X.assign(constante=1.0)[["constante", *X.columns]]


def ajustar_fgls(y, X, order, tolerancia=0.1, max_vueltas=20):
    """Devuelve (resultados de SARIMAX con los parámetros FGLS fijos, informe).

    y: demanda con las horas COVID vacías (NaN); X: exógenas sin constante.
    Ambas con índice horario regular (freq="h") y sin zona horaria. Para prever con
    los resultados, las exógenas también llevan la constante: con_constante(X).
    """
    X = con_constante(X)
    ok = y.notna()
    Xv = X.to_numpy()

    # Paso 1: regresión por mínimos cuadrados ordinarios, sin las horas COVID
    beta = np.linalg.lstsq(Xv[ok], y[ok].to_numpy(), rcond=None)[0]
    arma, llf_prev, convergio = None, -np.inf, False
    for vuelta in range(1, max_vueltas + 1):
        # Paso 2: ARMA de los residuos (el filtro de Kalman salta las horas vacías)
        res_arma = SARIMAX(y - Xv @ beta, order=order).fit(disp=False, start_params=arma)
        arma = res_arma.params

        # Paso 3: mínimos cuadrados generalizados con las series blanqueadas por el ARMA
        def blanquear(z):
            f = res_arma.apply(z.where(ok)).filter_results
            return f.forecasts_error[0] / np.sqrt(f.forecasts_error_cov[0, 0])

        yb = blanquear(y)
        Xb = np.column_stack([blanquear(X[c]) for c in X.columns])
        validas = ~np.isnan(yb)
        beta = np.linalg.lstsq(Xb[validas], yb[validas], rcond=None)[0]

        # Paso 4: regla de parada sobre la log-verosimilitud del modelo completo
        llf = SARIMAX(y - Xv @ beta, order=order).filter(arma).llf
        if abs(llf - llf_prev) < tolerancia:
            convergio = True
            break
        llf_prev = llf

    # Modelo completo de statsmodels con los parámetros FGLS fijos (previsión e intervalos)
    modelo = SARIMAX(y, exog=X, order=order)
    assert modelo.param_names == [*X.columns, *arma.index]  # mismo orden de parámetros
    params = np.concatenate([beta, arma.to_numpy()])
    resultados = modelo.filter(params)
    informe = {"vueltas": vuelta, "convergio": convergio, "llf": resultados.llf}
    return resultados, informe
