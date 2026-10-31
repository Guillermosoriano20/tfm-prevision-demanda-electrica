"""Capa probabilística: rejilla de cuantiles, intervalos del ingenuo, métricas, ACI y bootstrap.

Implementa docs/protocolo_oe5.md, apartado 14 (fijado con commit antes de escribir este
código). Módulo sin número porque se importa, como validacion.py. No lee datos por su
cuenta: recibe tablas y arreglos, de modo que sirve igual para 2023 (validación) y, más
adelante, para la evaluación final de 2024.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

import validacion as v

SEMILLA = 42
NIVELES = np.round(np.arange(1, 20) / 20, 2)                  # τ = 0,05 … 0,95 (14.4.1)
COLUMNAS_Q = [f"q{round(100 * t):02d}" for t in NIVELES]       # q05 … q95
ALFA = 0.10                                                    # intervalo del 90 %
BORDES_BLOQUE = (18, 24, 30)                                   # B1 12-17, B2 18-23, B3 24-29, B4 30-36
N_BLOQUES = 4
VENTANA = 336                                                  # W de ACI (D5)
GAMMAS = (0.001, 0.005, 0.01, 0.02)                            # candidatos (14.6.7)
REPLICAS = 1000                                                # bootstrap de días (D10)
DIAS_ARRANQUE_2023 = 57                                        # 1/1 a 26/2 (D6)


# ---------------------------------------------------------------- orígenes y bloques ---
def origenes_anio(anio):
    """Igual que validacion.generar_origenes, para cualquier año (el ingenuo usa 2017-2022)."""
    dias = pd.date_range(f"{anio}-01-01", f"{anio}-12-31", freq="D")
    tabla = pd.DataFrame({"fecha_objetivo": dias.date})
    tabla["origen_utc"] = (dias - v.UN_DIA + v.HORA_ORIGEN * v.UNA_HORA
                           ).tz_localize(v.ZONA).tz_convert("UTC")
    tabla["inicio_utc"] = dias.tz_localize(v.ZONA).tz_convert("UTC")
    tabla["fin_utc"] = (dias + v.UN_DIA).tz_localize(v.ZONA).tz_convert("UTC")
    tabla["n_horas"] = ((tabla["fin_utc"] - tabla["inicio_utc"]) / v.UNA_HORA).astype(int)
    return tabla


def anticipacion_y_bloque(tabla, origenes):
    """Anticipación (horas desde el origen) y bloque 0-3 de cada fila de la tabla común."""
    origen = tabla["fecha_objetivo"].map(origenes.set_index("fecha_objetivo")["origen_utc"])
    assert origen.notna().all(), "fecha objetivo sin origen"
    antic = ((tabla["hora_utc"] - origen) / v.UNA_HORA).to_numpy()
    assert antic.min() == 12 and antic.max() <= 36, "anticipación fuera de 12-36 h"
    return antic, np.digitize(antic, BORDES_BLOQUE)


def a_ns(instantes):
    """Instantes (serie, índice o arreglo con zona) como datetime64[ns] UTC sin zona."""
    return pd.DatetimeIndex(instantes).tz_convert("UTC").tz_localize(None).as_unit("ns").to_numpy()


# ----------------------------------------------------------------------- ingenuo (D3, D9)
def errores_ingenuo(datos, prever_ingenuo, anios):
    """Errores real − previsto del ingenuo en los años dados, sin COVID (14.3.1).

    Salen del 8/1/2017 (los siete primeros días no tienen semana anterior) y sin los días
    cuyo objetivo o cuya fuente (7 días antes) tocan `excluir_covid`.
    """
    covid = set(datos.loc[datos["excluir_covid"], "fecha_local"].unique())
    errores, bloques = [], []
    for anio in anios:
        for origen in origenes_anio(anio).itertuples():
            dia = pd.Timestamp(origen.fecha_objetivo)
            if dia < pd.Timestamp("2017-01-08") or dia in covid or dia - 7 * v.UN_DIA in covid:
                continue
            p = v.recortar(prever_ingenuo(datos, origen), origen)
            real = datos.loc[p.index, "demanda_mw"].to_numpy()
            antic = ((p.index - origen.origen_utc) / v.UNA_HORA).to_numpy()
            errores.append(real - p.to_numpy())
            bloques.append(np.digitize(antic, BORDES_BLOQUE))
    errores, bloques = np.concatenate(errores), np.concatenate(bloques)
    # El dataset empieza a las 01:00 locales del 1/1/2017: la hora 00 de ese día no existe y el
    # ingenuo del 8/1/2017 a las 00:00 queda vacío. Se descarta (solo esa hora) y se comprueba.
    validos = np.isfinite(errores)
    assert (~validos).sum() <= 1, f"{(~validos).sum()} errores del ingenuo no finitos"
    return errores[validos], bloques[validos]


def cuantiles_por_bloque(errores, bloques):
    """Cuantil empírico (interpolación lineal) de los errores, por bloque y nivel: (4, 19)."""
    return np.array([np.quantile(errores[bloques == b], NIVELES) for b in range(N_BLOQUES)])


# ---------------------------------------------------------------------------- métricas ---
def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def crps19(y, Q):
    """CRPS₁₉ = (2/19)·Σ_k ρ_τk … con paso 0,05: (2/20)·Σ_k ρ_τk (14.5.6). Por hora."""
    return 0.10 * pinball(y[:, None], Q, NIVELES[None, :]).sum(axis=1)


def puntuacion_intervalo(y, inferior, superior, alfa=ALFA):
    """IS (Gneiting y Raftery, 2007) por hora: amplitud más (2/α)·exceso por cada lado."""
    return ((superior - inferior) + (2 / alfa) * np.maximum(inferior - y, 0)
            + (2 / alfa) * np.maximum(y - superior, 0))


def fiabilidad(y, Q):
    """Frecuencia observada de real ≤ q_τk para cada τ de la rejilla."""
    return (y[:, None] <= Q).mean(axis=0)


def media_bootstrap(valores, dias, replicas=REPLICAS, semilla=SEMILLA):
    """Réplicas bootstrap de la media por horas, remuestreando días enteros (D10)."""
    codigos = pd.factorize(pd.Series(dias))[0]
    n = codigos.max() + 1
    suma = np.bincount(codigos, weights=np.asarray(valores, float), minlength=n)
    cuenta = np.bincount(codigos, minlength=n)
    idx = np.random.default_rng(semilla).integers(0, n, size=(replicas, n))
    return suma[idx].sum(axis=1) / cuenta[idx].sum(axis=1)


def ic95(valores, dias):
    """(media, límite inferior, límite superior) del IC del 95 % por bootstrap de días."""
    r = media_bootstrap(valores, dias)
    return float(np.mean(valores)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def reordenar(Q):
    """Reordenación monótona por hora (14.4.5). Devuelve la rejilla y las horas cambiadas."""
    ordenada = np.sort(Q, axis=1)
    return ordenada, int((ordenada != Q).any(axis=1).sum())


def metricas_intervalo(y, inferior, superior, dias, bloque, mes):
    """Cobertura, fallos, anchura e IS de un intervalo del 90 % (14.5.3-14.5.5)."""
    dentro = (inferior <= y) & (y <= superior)
    ancho = superior - inferior
    is_h = puntuacion_intervalo(y, inferior, superior)
    cob, cob_lo, cob_hi = ic95(dentro.astype(float), dias)
    por_bloque = [dentro[bloque == b].mean() for b in range(N_BLOQUES)]
    por_mes = [dentro[mes == m].mean() if (mes == m).any() else np.nan for m in range(1, 13)]
    por_debajo, por_encima = (y < inferior).mean(), (y > superior).mean()
    return {
        "cobertura": cob, "cob_ic_inf": cob_lo, "cob_ic_sup": cob_hi,
        "fallo_debajo": por_debajo, "fallo_encima": por_encima,
        "cociente_fallos": por_debajo / por_encima if por_encima > 0 else np.inf,
        **{f"cob_B{b + 1}": x for b, x in enumerate(por_bloque)},
        **{f"cob_mes{m:02d}": x for m, x in enumerate(por_mes, 1)},
        "ancho_medio": ancho.mean(), "ancho_mediana": np.median(ancho),
        "ancho_relativo": (ancho / y).mean(), "IS": is_h.mean(),
        "pinball05": pinball(y, inferior, 0.05).mean(), "pinball95": pinball(y, superior, 0.95).mean(),
    }


# ------------------------------------------------------------------------------- ACI ---
class BuscaPrevisiones(RegressorMixin, BaseEstimator):
    """«Estimador» de MAPIE que devuelve la previsión guardada de cada fila (14.6.2).

    La entrada X es el número de fila; no se entrena nada. Así MAPIE calcula las
    puntuaciones de conformidad con las previsiones oficiales de cada modelo.
    """

    def __init__(self, prevision=None):
        self.prevision = prevision

    def fit(self, X, y=None):
        self.n_features_in_ = 1
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.prevision[np.asarray(X, dtype=float).astype(int).ravel()]

    def __sklearn_is_fitted__(self):
        return True


def actualizar_alfa(alfa_t, alfa0, gamma, fallo):
    """Recursión de ACI idéntica a MAPIE.adapt_conformal_inference (D4, 14.6.5)."""
    return float(np.clip(alfa_t + gamma * (alfa0 - float(fallo)), 0, 1))


def fallo_intervalo(y, inferior, superior):
    """Criterio estricto de MAPIE: dentro si inferior < y < superior."""
    return not ((y > inferior) and (y < superior))


def motor_aci(hora_utc, origen_utc, real, prevision, bloque, dia, configuraciones,
              primer_dia, hasta_dia=None, ventana=VENTANA, fuga=False):
    """ACI por bloques con MAPIE (14.6.4-14.6.5).

    Todas las entradas son arreglos por fila, en orden cronológico estricto:
      hora_utc, origen_utc : datetime64[ns] (origen = el del día objetivo de la fila);
      real, prevision      : MW; bloque: 0-3; dia: entero no decreciente (0, 1, 2 …).
    configuraciones: lista de (alfa0, gamma); cada una lleva su propio α_t por bloque.
    primer_dia: primer día del que se emiten intervalos (los anteriores solo aportan
    residuos). hasta_dia: se detiene antes de ese día (pruebas). fuga=True es la variante
    del control positivo (14.7.3): la ventana incluye las horas del propio día objetivo.
    Devuelve dict con inferior, superior, alfa_emitido (n_config × n_filas, NaN si no se
    emite) y horas_infinitas (D8) por configuración.
    """
    from mapie.regression import TimeSeriesRegressor

    n = len(real)
    hora_fin = hora_utc + np.timedelta64(1, "h")
    assert (np.diff(hora_utc.astype("int64")) > 0).all(), "horas no estrictamente crecientes"
    alfas0 = np.array([round(a, 2) for a, _ in configuraciones])
    gammas = np.array([g for _, g in configuraciones])
    n_cfg = len(configuraciones)
    inicio = np.flatnonzero(np.r_[True, dia[1:] != dia[:-1]])
    fin = np.r_[inicio[1:], n]
    dias_ids = dia[inicio]
    assert primer_dia in dias_ids
    ultimo = len(inicio) if hasta_dia is None else int(np.searchsorted(dias_ids, hasta_dia))
    primera_fila_emitida = inicio[int(np.searchsorted(dias_ids, primer_dia))]
    filas_bloque = [np.flatnonzero(bloque == b) for b in range(N_BLOQUES)]

    inferior = np.full((n_cfg, n), np.nan)
    superior = np.full((n_cfg, n), np.nan)
    alfa_emitido = np.full((n_cfg, n), np.nan)
    alfa = np.tile(alfas0[:, None], (1, N_BLOQUES))
    infinitas = np.zeros(n_cfg, dtype=int)
    visto = 0                                   # filas con retroalimentación ya aplicada
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")         # avisos internos de MAPIE/sklearn
        for i in range(int(np.searchsorted(dias_ids, primer_dia)), ultimo):
            n_obs = int(np.searchsorted(hora_fin, origen_utc[inicio[i]], side="right"))
            # 1. Retroalimentación: filas emitidas cuya hora ya terminó (14.6.5)
            for r in range(max(visto, primera_fila_emitida), n_obs):
                b = bloque[r]
                fallo = ~((real[r] > inferior[:, r]) & (real[r] < superior[:, r]))
                alfa[:, b] = np.clip(alfa[:, b] + gammas * (alfas0 - fallo), 0, 1)
            visto = max(visto, n_obs)
            n_ventana = fin[i] if fuga else n_obs
            # 2. Un conjunto de puntuaciones por bloque: las W últimas observadas
            for b in range(N_BLOQUES):
                nuevas = np.arange(inicio[i], fin[i])[bloque[inicio[i]:fin[i]] == b]
                k = int(np.searchsorted(filas_bloque[b], n_ventana, side="left"))
                if not fuga and len(nuevas):
                    assert filas_bloque[b][k - 1] < nuevas[0]
                ven = filas_bloque[b][max(0, k - ventana):k]
                assert len(ven) == ventana, f"ventana incompleta (bloque {b}, día {dias_ids[i]})"
                m = TimeSeriesRegressor(estimator=BuscaPrevisiones(prevision), method="aci",
                                        cv="prefit")
                m.fit(ven[:, None].astype(float), real[ven])
                residuos = real[ven] - prevision[ven]
                for c in range(n_cfg):
                    m.current_alpha = {float(alfas0[c]): float(alfa[c, b])}
                    _, lim = m.predict(nuevas[:, None].astype(float),
                                       confidence_level=1 - alfas0[c], allow_infinite_bounds=True)
                    lo, hi = lim[:, 0, 0].copy(), lim[:, 1, 0].copy()
                    mal_lo, mal_hi = ~np.isfinite(lo), ~np.isfinite(hi)       # D8
                    lo[mal_lo] = prevision[nuevas][mal_lo] + residuos.min()
                    hi[mal_hi] = prevision[nuevas][mal_hi] + residuos.max()
                    infinitas[c] += int((mal_lo | mal_hi).sum())
                    inferior[c, nuevas], superior[c, nuevas] = lo, hi
                    alfa_emitido[c, nuevas] = alfa[c, b]
    return {"inferior": inferior, "superior": superior, "alfa_emitido": alfa_emitido,
            "horas_infinitas": infinitas}


def mediana_ventana(hora_utc, origen_utc, real, prevision, bloque, dia, primer_dia,
                    ventana=VENTANA):
    """Cuantil 0,5 sin adaptación: previsión + mediana de la ventana de residuos (14.6.8)."""
    n = len(real)
    hora_fin = hora_utc + np.timedelta64(1, "h")
    inicio = np.flatnonzero(np.r_[True, dia[1:] != dia[:-1]])
    fin = np.r_[inicio[1:], n]
    filas_bloque = [np.flatnonzero(bloque == b) for b in range(N_BLOQUES)]
    mediana = np.full(n, np.nan)
    for i in range(int(np.searchsorted(dia[inicio], primer_dia)), len(inicio)):
        n_obs = int(np.searchsorted(hora_fin, origen_utc[inicio[i]], side="right"))
        for b in range(N_BLOQUES):
            k = int(np.searchsorted(filas_bloque[b], n_obs, side="left"))
            ven = filas_bloque[b][k - ventana:k]
            filas = np.arange(inicio[i], fin[i])
            filas = filas[bloque[filas] == b]
            mediana[filas] = prevision[filas] + np.median(real[ven] - prevision[ven])
    return mediana


def cuantiles_de_aci(hora_utc, origen_utc, real, prevision, bloque, dia, primer_dia, gamma):
    """Rejilla de 19 cuantiles con ACI en los 9 niveles centrales (14.6.8).

    Devuelve (rejilla n×19 ya reordenada, horas reordenadas, horas infinitas por nivel).
    Las filas anteriores a `primer_dia` quedan con NaN.
    """
    centrales = np.round(np.arange(1, 10) / 10, 1)                 # 0,1 … 0,9
    configuraciones = [(round(1 - c, 2), gamma) for c in centrales]
    r = motor_aci(hora_utc, origen_utc, real, prevision, bloque, dia, configuraciones, primer_dia)
    Q = np.full((len(real), len(NIVELES)), np.nan)
    for c, nivel in enumerate(centrales):
        Q[:, int(np.argmin(abs(NIVELES - round((1 - nivel) / 2, 2))))] = r["inferior"][c]
        Q[:, int(np.argmin(abs(NIVELES - round((1 + nivel) / 2, 2))))] = r["superior"][c]
    Q[:, int(np.argmin(abs(NIVELES - 0.5)))] = mediana_ventana(
        hora_utc, origen_utc, real, prevision, bloque, dia, primer_dia)
    emitido = ~np.isnan(Q[:, 0])
    Q[emitido], reordenadas = reordenar(Q[emitido])
    return Q, reordenadas, r["horas_infinitas"]


# ------------------------------------------------------------------ controles (14.7) ---
def sintetico(dias=207, semilla=1, salto=None, sigma=100.0):
    """Serie sintética con la estructura de la tabla común: errores independientes de escala
    conocida y, si `salto=(día, factor)`, un cambio de escala a partir de ese día."""
    rng = np.random.default_rng(semilla)
    n = dias * 24
    hora = np.datetime64("2023-01-01T00:00") + np.arange(n) * np.timedelta64(1, "h")
    dia = np.repeat(np.arange(dias), 24)
    origen = (hora.astype("datetime64[D]")[::24] - np.timedelta64(12, "h"))[dia]
    antic = ((hora - origen) / np.timedelta64(1, "h")).astype(int)
    escala = np.full(n, sigma)
    if salto:
        escala[salto[0] * 24:] *= salto[1]
    prevision = 1000 + 50 * np.sin(np.arange(n) / 24 * 2 * np.pi / 7)
    real = prevision + rng.normal(0, 1, n) * escala
    return {"hora_utc": hora.astype("datetime64[ns]"), "origen_utc": origen.astype("datetime64[ns]"),
            "real": real, "prevision": prevision, "bloque": np.digitize(antic, BORDES_BLOQUE),
            "dia": dia}


def control_recursion_mapie(gammas=(0.0, 0.005, 0.02), n=3000, inicio=500, semilla=0):
    """14.7.2: con retroalimentación inmediata, la recursión propia = adapt_conformal_inference."""
    from mapie.regression import TimeSeriesRegressor
    rng = np.random.default_rng(semilla)
    y = rng.normal(0, 1, n) * np.where(np.arange(n) > n // 2, 2.0, 1.0) + 0.5
    pred, X = np.zeros(n), np.arange(n).reshape(-1, 1).astype(float)
    iguales = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for g in gammas:
            m = TimeSeriesRegressor(estimator=BuscaPrevisiones(pred), method="aci", cv="prefit")
            m.fit(X[:inicio], y[:inicio])
            de_mapie = []
            for i in range(inicio, n):
                m.predict(X[i:i + 1], confidence_level=0.9, allow_infinite_bounds=True)
                m.adapt_conformal_inference(X[i:i + 1], y[i:i + 1], gamma=g, confidence_level=0.9)
                de_mapie.append(m.current_alpha[0.1])
            m = TimeSeriesRegressor(estimator=BuscaPrevisiones(pred), method="aci", cv="prefit")
            m.fit(X[:inicio], y[:inicio])
            a, propia = 0.1, []
            for i in range(inicio, n):
                m.current_alpha = {0.1: a}
                _, b = m.predict(X[i:i + 1], confidence_level=0.9, allow_infinite_bounds=True)
                a = actualizar_alfa(a, 0.1, g, fallo_intervalo(y[i], b[0, 0, 0], b[0, 1, 0]))
                propia.append(a)
            iguales[g] = bool(np.array_equal(de_mapie, propia))
    return iguales


def control_causalidad(entradas, primer_dia, cortes, gamma=0.01):
    """14.7.3: la demanda posterior al corte no cambia ningún intervalo emitido con origen ≤ corte.

    `entradas`: dict con hora_utc, origen_utc, real, prevision, bloque, dia (por fila).
    Para cada día de corte k: el origen del día k es el corte; se suman 10.000 MW a toda la
    demanda desde ese instante. Debe cumplirse: (a) los intervalos de los días ≤ k son
    idénticos; (b) los del día k+1 sí cambian (la prueba tiene potencia); (c) la variante
    con fuga (ventana con el propio día objetivo) cambia ya los del día k (control positivo).
    Devuelve una lista de (k, a, b, c) booleanos.
    """
    e = entradas
    cfg = [(ALFA, gamma)]
    resultados = []
    for k in cortes:
        corte = e["origen_utc"][np.flatnonzero(e["dia"] == k)[0]]
        alterada = np.where(e["hora_utc"] >= corte, e["real"] + 10_000, e["real"])

        def correr(real, hasta, fuga=False):
            return motor_aci(e["hora_utc"], e["origen_utc"], real, e["prevision"], e["bloque"],
                             e["dia"], cfg, primer_dia, hasta_dia=hasta, fuga=fuga)

        def iguales(a, b, dias):
            f = np.isin(e["dia"], dias)
            return bool(np.array_equal(a["inferior"][:, f], b["inferior"][:, f], equal_nan=True)
                        and np.array_equal(a["superior"][:, f], b["superior"][:, f], equal_nan=True))

        base, alt = correr(e["real"], k + 2), correr(alterada, k + 2)
        hasta_k = np.arange(primer_dia, k + 1)
        fuga_base, fuga_alt = correr(e["real"], k + 1, True), correr(alterada, k + 1, True)
        resultados.append((k, iguales(base, alt, hasta_k), not iguales(base, alt, [k + 1]),
                           not iguales(fuga_base, fuga_alt, [k])))
    return resultados


def control_sintetico(gammas=GAMMAS, tolerancia=0.02):
    """14.7.4: cobertura 0,90 ± 0,02 con errores independientes y recuperación tras un salto."""
    sin_salto = sintetico(dias=57 + 250, semilla=11)
    con_salto = sintetico(dias=57 + 250, semilla=12, salto=(57 + 120, 2.0))
    cfg = [(ALFA, g) for g in gammas]
    salida = {}
    for nombre, e in (("sin_salto", sin_salto), ("salto", con_salto)):
        r = motor_aci(e["hora_utc"], e["origen_utc"], e["real"], e["prevision"], e["bloque"],
                      e["dia"], cfg, primer_dia=DIAS_ARRANQUE_2023)
        emitido = ~np.isnan(r["inferior"][0])
        cola = emitido & (e["dia"] >= 57 + 120 + 60)          # 60 días después del salto
        for c, g in enumerate(gammas):
            dentro = (r["inferior"][c] <= e["real"]) & (e["real"] <= r["superior"][c])
            salida[(nombre, g)] = (dentro[emitido].mean(), dentro[cola].mean())
    ok = all(abs(salida[("sin_salto", g)][0] - 0.90) <= tolerancia for g in gammas) and all(
        abs(salida[("salto", g)][1] - 0.90) <= tolerancia for g in gammas)
    return ok, salida
