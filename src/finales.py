"""Evaluación final en 2024: calendario de reajustes, componentes, ensamblado y contrastes.

Implementa docs/protocolo_oe5.md, apartado 15 (fijado con commit antes de escribir este
código). Módulo sin número porque se importa, como validacion.py y probabilistica.py. Lo
usan 28_prevision_final_2024.py (genera las previsiones, sin imprimir ninguna métrica de
error) y 29_evaluacion_final_2024.py (la lectura única).

Reutiliza sin modificar los módulos cerrados: validacion, lightgbm_modelo, nhits_modelo,
nhits_rejilla, probabilistica, exogenas y sarimax_fgls, y los scripts 15 y 26.
"""
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

import lightgbm_modelo as lm
import nhits_modelo as nm
import nhits_rejilla as nr
import probabilistica as P
import validacion as v

ARNES = os.environ.get("TFM_ARNES") == "1"        # solo en el entorno de pruebas del asistente
DIR_INTERIM = v.RAIZ / "data" / "interim" / "final_2024"
DIR_RESULTADOS = v.RAIZ / "data" / "resultados" / "final"
MODELOS = ("ingenuo", "sarimax", "lightgbm", "nhits")
COMPONENTES = ("ingenuo", "sarimax", "lightgbm", "nhits", "nhits19")
NOMBRE = {"ingenuo": "Ingenuo", "sarimax": "SARIMAX", "lightgbm": "LightGBM", "nhits": "N-HiTS"}
ORDEN_SARIMAX = (5, 0, 1)                         # ARMA(5,1) con `nivel` y retardos (10.10)
TRIMESTRALES = (1, 4, 7, 10)                      # reajustes que se mantienen con el recorte (15.10.4)
RETARDO_DM = 7                                    # Newey-West (15.6)
SEMANA = ("2024-05-13", "2024-05-19")             # semana de la figura, fijada en 15.5


def aplicar_arnes():
    """Presupuestos reducidos, solo para probar el código (15.12.3). Nunca en la ejecución real."""
    lm.MAX_ARBOLES, lm.PACIENCIA = 25, 5
    nm.P_MAX, nm.CONTROL_CADA, nm.PACIENCIA = 40, 20, 2


_MODULOS = {}


def cargar_script(nombre):
    """Importa un script con número en el nombre (no se puede con `import`); una sola vez."""
    if nombre in _MODULOS:
        return _MODULOS[nombre]
    ruta = Path(__file__).resolve().parent / nombre
    especificacion = importlib.util.spec_from_file_location("s_" + nombre[:2], ruta)
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    _MODULOS[nombre] = modulo
    return modulo


# ------------------------------------------------------------------------ calendario ---
def calendario():
    """Los 12 reajustes de 15.3: origen del reajuste y días objetivo que prevé cada uno."""
    o = v.generar_origenes("prueba")
    assert len(o) == 366 and int(o["n_horas"].sum()) == 8784
    o["mes"] = pd.to_datetime(o["fecha_objetivo"]).dt.month
    filas = [{"k": int(k), "fin": g["origen_utc"].iloc[0], "desde": g["fecha_objetivo"].iloc[0],
              "hasta": g["fecha_objetivo"].iloc[-1], "n_dias": len(g)} for k, g in o.groupby("mes")]
    cal = pd.DataFrame(filas)
    assert cal["k"].tolist() == list(range(1, 13))
    assert cal["n_dias"].tolist() == [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    assert str(cal["fin"].iloc[0].tz_convert(v.ZONA))[:16] == "2023-12-31 12:00"
    assert o.loc[o["n_horas"] == 23, "fecha_objetivo"].astype(str).tolist() == ["2024-03-31"]
    assert o.loc[o["n_horas"] == 25, "fecha_objetivo"].astype(str).tolist() == ["2024-10-27"]
    return cal, o


def origenes_mes(o, k):
    return o[o["mes"] == k].reset_index(drop=True)


def rutas(comp, k):
    return DIR_INTERIM / f"{comp}_{k:02d}.parquet", DIR_INTERIM / f"{comp}_{k:02d}.json"


def hecho(comp, k):
    return all(r.exists() for r in rutas(comp, k))


def guardar_mes(comp, k, tabla, meta):
    DIR_INTERIM.mkdir(parents=True, exist_ok=True)
    pq, js = rutas(comp, k)
    tabla.to_parquet(pq, index=False)
    js.write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")   # el último: marca «hecho»


def leer_meta(comp, k):
    return json.loads(rutas(comp, k)[1].read_text(encoding="utf-8"))


# ------------------------------------------------------------------ utilidades comunes ---
def minutos(t0):
    return (time.perf_counter() - t0) / 60


def sin_fuga(prever, datos, origenes, nombre):
    """Prueba de fuga reforzada (13): demanda posterior al origen borrada. 0 días o se detiene."""
    t0 = time.perf_counter()
    fugas = v.comprobar_fuga(prever, datos, origenes, modo="borrar")
    print(f"    fuga de la previsión ({nombre}): {len(fugas)} días con fuga ({minutos(t0):.1f} min)", flush=True)
    assert not fugas, f"FUGA DETECTADA en {nombre}: se detiene todo ({fugas[:3]})"
    return len(fugas)


def unir(cache, origenes, datos):
    """Une las previsiones del mes recortadas a su día objetivo: (previsiones, base común)."""
    trozos, fechas = [], []
    for o in origenes.itertuples():
        p = v.recortar(cache[o.fecha_objetivo], o)
        trozos.append(p if isinstance(p, pd.DataFrame) else p.to_frame("prevision"))
        fechas += [o.fecha_objetivo] * len(p)
    t = pd.concat(trozos)
    base = pd.DataFrame({"hora_utc": t.index, "fecha_objetivo": fechas,
                         "real": datos.loc[t.index, "demanda_mw"].to_numpy()})
    assert len(base) == int(origenes["n_horas"].sum())
    return t.reset_index(drop=True), base


def bloques_de(base, origenes):
    return P.anticipacion_y_bloque(base, origenes)[1]


def datos_alterados(datos, fin):
    """Copia con +10.000 MW en toda la demanda posterior al origen del reajuste (15.7.3)."""
    alt = datos.copy()
    alt.loc[alt.index >= fin, "demanda_mw"] += 10_000
    return alt


# ----------------------------------------------------------------------------- ingenuo ---
def errores_ingenuo_fechados(datos):
    """Errores del ingenuo 2017-2024 sin COVID, con el instante en que termina su día objetivo."""
    modulo = cargar_script("15_ingenuo_estacional.py")
    covid = set(datos.loc[datos["excluir_covid"], "fecha_local"].unique())
    err, blo, fin_dia = [], [], []
    for anio in range(2017, 2025):
        for o in P.origenes_anio(anio).itertuples():
            dia = pd.Timestamp(o.fecha_objetivo)
            if dia < pd.Timestamp("2017-01-08") or dia in covid or dia - 7 * v.UN_DIA in covid:
                continue
            p = v.recortar(modulo.prever_ingenuo(datos, o), o)
            e = datos.loc[p.index, "demanda_mw"].to_numpy() - p.to_numpy()
            antic = ((p.index - o.origen_utc) / v.UNA_HORA).to_numpy()
            err.append(e)
            blo.append(np.digitize(antic, P.BORDES_BLOQUE))
            fin_dia.append(np.full(len(e), o.fin_utc.value))
    err, blo, fin_dia = np.concatenate(err), np.concatenate(blo), np.concatenate(fin_dia)
    assert (~np.isfinite(err)).sum() <= 1
    return err, blo, fin_dia


def cuantiles_ingenuo(err, blo, fin_dia, fin):
    """Cuantiles por bloque con los errores de los días que terminan antes del reajuste (F7)."""
    usar = (fin_dia <= fin.value) & np.isfinite(err)
    return P.cuantiles_por_bloque(err[usar], blo[usar]), int(usar.sum())


def _s26():
    return cargar_script("26_cuantiles_validacion.py")


def componente_ingenuo(datos, k, fin, origenes, ctx):
    t0 = time.perf_counter()
    modulo = cargar_script("15_ingenuo_estacional.py")
    err, blo, fdia = ctx["errores"]
    Qb, n = cuantiles_ingenuo(err, blo, fdia, fin)
    if k in (1, 7):                                   # prueba de fuga del reajuste (15.7.3)
        Qb2, n2 = cuantiles_ingenuo(*errores_ingenuo_fechados(datos_alterados(datos, fin)), fin)
        assert n == n2 and np.array_equal(Qb, Qb2), "FUGA en los cuantiles del ingenuo"
        print("    fuga del reajuste (ingenuo): cuantiles idénticos con la demanda posterior alterada", flush=True)
    prever, cache = _s26().con_cache(modulo.prever_ingenuo, datos)
    fugas = sin_fuga(prever, datos, origenes, "ingenuo")
    t, base = unir(cache, origenes, datos)
    Q = t["prevision"].to_numpy()[:, None] + Qb[bloques_de(base, origenes)]
    tabla = pd.concat([base, t[["prevision"]], pd.DataFrame(Q, columns=P.COLUMNAS_Q)], axis=1)
    return tabla, {"n_errores": n, "fugas": fugas, "minutos": minutos(t0)}


# ----------------------------------------------------------------------------- SARIMAX ---
def ajustar_sarimax_final(datos, fin):
    """FGLS desde cero con la ventana hasta `fin` (F8). Se detiene si no converge."""
    s26 = _s26()
    preparar = s26.preparar_sarimax(datos)
    y, Xm = preparar(datos)
    corte = fin.tz_localize(None)
    extra = {"max_vueltas": 1} if ARNES else {}
    res, informe = s26.ajustar_fgls(y[y.index < corte], Xm[Xm.index < corte],
                                    order=ORDEN_SARIMAX, **extra)
    if not ARNES:
        assert informe["convergio"], "el FGLS no convergió: se detiene todo (F8)"
    return res, preparar, informe


def componente_sarimax(datos, k, fin, origenes, ctx):
    t0 = time.perf_counter()
    s26 = _s26()
    res, preparar, informe = ajustar_sarimax_final(datos, fin)
    print(f"    FGLS: {informe['vueltas']} vueltas, convergió {informe['convergio']} ({minutos(t0):.1f} min)", flush=True)
    t_prueba = 0.0
    if k in (1, 7):                                   # prueba de fuga del reajuste (15.7.3)
        t1 = time.perf_counter()
        res2, _, _ = ajustar_sarimax_final(datos_alterados(datos, fin), fin)
        assert np.array_equal(res.params, res2.params), "FUGA en el ajuste del SARIMAX"
        t_prueba = minutos(t1)
        print("    fuga del reajuste (SARIMAX): parámetros idénticos con la demanda posterior alterada", flush=True)
    prever, cache = s26.con_cache(s26.hacer_prever_sarimax(res, preparar), datos)
    fugas = sin_fuga(prever, datos, origenes, "SARIMAX: media y error típico")
    t, base = unir(cache, origenes, datos)
    se = t["se"].to_numpy()
    assert (se > 0).all() and np.isfinite(se).all()
    Q = t["media"].to_numpy()[:, None] + norm.ppf(P.NIVELES)[None, :] * se[:, None]
    tabla = pd.concat([base, t[["media"]].rename(columns={"media": "prevision"}), t[["se"]],
                       pd.DataFrame(Q, columns=P.COLUMNAS_Q)], axis=1)
    arma = {n: float(x) for n, x in res.params.items() if n.startswith(("ar.", "ma.", "sigma2"))}
    return tabla, {"vueltas": informe["vueltas"], "convergio": bool(informe["convergio"]),
                   "arma": arma, "fugas": fugas, "minutos": minutos(t0),
                   "minutos_prueba_reajuste": t_prueba}


# ---------------------------------------------------------------------------- LightGBM ---
def ruta_modelo_q(tau, k):
    return DIR_INTERIM / "modelos" / f"lgb_q{round(100 * tau):02d}_{k:02d}.txt"


def prever_lightgbm(m_punto, modelos):
    """prever(datos, origen): previsión puntual y los 19 cuantiles de las horas del día D+1."""
    def prever(d, origen):
        horas = pd.date_range(origen.inicio_utc, origen.fin_utc, freq="h",
                              inclusive="left").as_unit(d.index.unit)
        X = lm.variables(d, horas, pd.DatetimeIndex([origen.origen_utc] * len(horas)))
        assert not X.isna().any().any(), f"variables ausentes el {origen.fecha_objetivo}"
        salida = pd.DataFrame({c: m.predict(X) for c, m in zip(P.COLUMNAS_Q, modelos)}, index=horas)
        salida.insert(0, "prevision", m_punto.predict(X))
        return salida
    return prever


def componente_lightgbm(datos, k, fin, origenes, ctx):
    t0 = time.perf_counter()
    parametros = _s26().parametros_c17()
    X, y = lm.filas_entrenamiento(datos, fin)
    recorte = ctx["recorte"]
    k_trim = max(j for j in TRIMESTRALES if j <= k)
    m_punto, n_punto = lm.entrenar(X, y, fin, parametros)
    t_prueba = 0.0
    if k in (1, 7):                                   # prueba de fuga del reajuste (15.7.3)
        t1 = time.perf_counter()
        alt = datos_alterados(datos, fin)
        X2, y2 = lm.filas_entrenamiento(alt, fin)
        assert X.equals(X2) and y.equals(y2), "FUGA en las filas de entrenamiento de LightGBM"
        for alpha in (None, 0.05, 0.95):
            a, _ = lm.entrenar(X, y, fin, parametros, alpha=alpha)
            b, _ = lm.entrenar(X2, y2, fin, parametros, alpha=alpha)
            assert a.model_to_string() == b.model_to_string(), f"FUGA en el modelo de LightGBM ({alpha})"
        t_prueba = minutos(t1)
        print("    fuga del reajuste (LightGBM): filas y modelos idénticos con la demanda posterior alterada", flush=True)
    modelos, arboles = [], []
    for tau in P.NIVELES:
        tau = float(tau)
        trimestral = recorte >= 2 and tau not in (0.05, 0.95)
        if trimestral and k not in TRIMESTRALES:
            ruta = ruta_modelo_q(tau, k_trim)
            assert ruta.exists(), f"falta {ruta.name}: empieza el recorte 2 en un reajuste trimestral"
            m = lgb.Booster(model_file=str(ruta))
            n = m.num_trees()
        else:
            m, n = lm.entrenar(X, y, fin, parametros, alpha=tau)
            if trimestral:
                ruta_modelo_q(tau, k).parent.mkdir(parents=True, exist_ok=True)
                m.save_model(str(ruta_modelo_q(tau, k)))
        modelos.append(m)
        arboles.append(int(n))
    print(f"    árboles: puntual {n_punto}; cuantil 0,05: {arboles[0]}; cuantil 0,95: {arboles[-1]}"
          f" ({minutos(t0):.1f} min)", flush=True)
    prever, cache = _s26().con_cache(prever_lightgbm(m_punto, modelos), datos)
    fugas = sin_fuga(prever, datos, origenes, "LightGBM: puntual y 19 cuantiles")
    t, base = unir(cache, origenes, datos)
    tabla = pd.concat([base, t], axis=1)
    return tabla, {"arboles_punto": int(n_punto), "arboles_q": arboles, "recorte": recorte,
                   "fugas": fugas, "minutos": minutos(t0), "minutos_prueba_reajuste": t_prueba}


# ----------------------------------------------------------------------------- N-HiTS ---
def hiper_nhits():
    """Hiperparámetros de c13 (12.6) en el formato de la librería."""
    c = nr.config_c13()
    resto = {k: x for k, x in c.items() if k not in ("input_size", "ancho", "escalas")}
    if ARNES:
        resto["windows_batch_size"] = 64
    return c["input_size"], nm.arquitectura(c["ancho"], c["escalas"], **resto)


def componente_nhits(datos, k, fin, origenes, ctx):
    """Red oficial (tres salidas): la mediana es la previsión puntual, q05 y q95 el intervalo."""
    t0 = time.perf_counter()
    L, hiper = hiper_nhits()
    nf, pasos, perdida = nm.entrenar(datos, fin, L, semilla=nm.SEMILLA, **hiper)
    print(f"    red oficial: {pasos} pasos, pérdida de control {perdida:.1f} ({minutos(t0):.1f} min)", flush=True)
    if ARNES and k == 1:                              # prueba completa del reajuste, solo con presupuesto reducido
        nf2, pasos2, _ = nm.entrenar(datos_alterados(datos, fin), fin, L, semilla=nm.SEMILLA, **hiper)
        o = origenes.iloc[0]
        a, b = nm.hacer_prever(nf)(datos, o), nm.hacer_prever(nf2)(datos, o)
        assert pasos == pasos2 and np.array_equal(a.to_numpy(), b.to_numpy()), "FUGA en el reajuste de N-HiTS"
        print("    fuga del reajuste (N-HiTS, prueba completa del arnés): previsiones idénticas", flush=True)
    prever, cache = _s26().con_cache(nm.hacer_prever(nf), datos)
    fugas = sin_fuga(prever, datos, origenes, "N-HiTS oficial")
    t, base = unir(cache, origenes, datos)
    tabla = pd.concat([base, t.rename(columns={"mediana": "prevision"})[["prevision", "q05", "q95"]]], axis=1)
    return tabla, {"pasos": int(pasos), "perdida_control": float(perdida), "fugas": fugas,
                   "minutos": minutos(t0)}


def componente_nhits19(datos, ks, fin, origenes_por_k, ctx):
    """Segunda red de 19 salidas. Con el recorte 1 una red sirve a varios meses (`ks`)."""
    t0 = time.perf_counter()
    L, hiper = hiper_nhits()
    nf, pasos, perdida = nr.entrenar(datos, fin, L, semilla=nm.SEMILLA, **hiper)
    print(f"    segunda red: {pasos} pasos, pérdida de control {perdida:.1f} ({minutos(t0):.1f} min)", flush=True)
    resultados = {}
    for j, k in enumerate(ks):
        prever, cache = _s26().con_cache(nr.hacer_prever(nf), datos)
        fugas = sin_fuga(prever, datos, origenes_por_k[k], f"N-HiTS 19 salidas, mes {k}")
        t, base = unir(cache, origenes_por_k[k], datos)
        resultados[k] = (pd.concat([base, t], axis=1),
                         {"pasos": int(pasos), "perdida_control": float(perdida), "fugas": fugas,
                          "modelo_de": ks[0], "recorte": ctx["recorte"],
                          "minutos": minutos(t0) if j == 0 else 0.0})
    return resultados


# --------------------------------------------------------------------------- ensamblado ---
def ensamblar(datos):
    """Une los 12 meses de cada componente y escribe los archivos en el formato común (15.3.5)."""
    cal, o = calendario()
    esperadas = pd.DatetimeIndex(np.concatenate([
        pd.date_range(r.inicio_utc, r.fin_utc, freq="h", inclusive="left").as_unit(datos.index.unit)
        for r in o.itertuples()]))
    assert len(esperadas) == 8784
    T = {c: pd.concat([pd.read_parquet(rutas(c, k)[0]) for k in range(1, 13)], ignore_index=True)
         for c in COMPONENTES}
    resumen = []
    for c, t in T.items():
        assert len(t) == 8784 and not t.isna().any().any(), f"{c}: filas o nulos"
        assert np.array_equal(P.a_ns(t["hora_utc"]), P.a_ns(esperadas)), f"{c}: horas distintas de las esperadas"
        real = datos["demanda_mw"].reindex(esperadas).to_numpy()
        assert np.array_equal(t["real"].to_numpy(), real), f"{c}: `real` distinta del dataset"
    v.DIR_PREDICCIONES.mkdir(parents=True, exist_ok=True)
    comunes = ["hora_utc", "fecha_objetivo", "real", "prevision"]
    for m in MODELOS:
        t = T[m]
        t[comunes].assign(modelo=m).to_parquet(v.DIR_PREDICCIONES / f"{m}_prueba.parquet", index=False)
        Q = T["nhits19"][P.COLUMNAS_Q].to_numpy() if m == "nhits" else t[P.COLUMNAS_Q].to_numpy()
        Q, reordenadas = P.reordenar(Q)
        rejilla = t[comunes].copy()
        for j, col in enumerate(P.COLUMNAS_Q):
            rejilla[col] = Q[:, j]
        rejilla.to_parquet(v.DIR_PREDICCIONES / f"{m}_cuantiles_prueba.parquet", index=False)
        fila = {"modelo": m, "horas_reordenadas": reordenadas}
        if m in ("lightgbm", "nhits"):                 # regla de cruce (11.2.5, 12.2.5)
            cruces = t["q05"] > t["q95"]
            inter = t[comunes].assign(inferior=np.where(cruces, t["q95"], t["q05"]),
                                      superior=np.where(cruces, t["q05"], t["q95"]), modelo=m)
            inter.to_parquet(v.DIR_PREDICCIONES / f"{m}_intervalos_prueba.parquet", index=False)
            fila["cruces"] = int(cruces.sum())
        resumen.append(fila)
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resumen).to_csv(DIR_RESULTADOS / "rejillas_resumen_2024.csv", index=False)
    return pd.DataFrame(resumen)


# ----------------------------------------------------------------- Diebold-Mariano (15.6) ---
def varianza_newey_west(d, m=RETARDO_DM):
    """Varianza de largo plazo con núcleo de Bartlett y autocovarianzas con divisor n."""
    d = np.asarray(d, float)
    n = len(d)
    c = d - d.mean()
    v_ = (c * c).sum() / n
    for j in range(1, m + 1):
        v_ += 2 * (1 - j / (m + 1)) * (c[j:] * c[:-j]).sum() / n
    return v_


def diebold_mariano(d, m=RETARDO_DM):
    """DM con corrección de Harvey, Leybourne y Newbold (h = 1) y t de n − 1 grados (bilateral)."""
    d = np.asarray(d, float)
    n = len(d)
    var = varianza_newey_west(d, m)
    if not var > 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var / n) * np.sqrt((n - 1) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), n - 1))


def holm(p):
    """p-valores ajustados por Holm (control del error de la familia)."""
    p = np.asarray(p, float)
    orden = np.argsort(p)
    ajustado = np.empty_like(p)
    maximo = 0.0
    for rango, i in enumerate(orden):
        maximo = max(maximo, min(1.0, (len(p) - rango) * p[i]))
        ajustado[i] = maximo
    return ajustado


def controles_dm(replicas=2000, n=366, semilla=42):
    """15.7.8: nivel, potencia y robustez a la autocorrelación del contraste. No usa 2024."""
    rng = np.random.default_rng(semilla)

    def tasa(media, phi):
        rechazos = 0
        for _ in range(replicas):
            e = rng.standard_normal(n + 50)
            d = np.zeros(n + 50)
            for t in range(1, n + 50):
                d[t] = phi * d[t - 1] + e[t]
            d = d[50:] * np.sqrt(1 - phi ** 2) + media
            rechazos += diebold_mariano(d)[1] < 0.05
        return rechazos / replicas
    return {"nivel": tasa(0.0, 0.0), "potencia": tasa(-0.3, 0.0), "ar1": tasa(0.0, 0.5)}


# --------------------------------------------------------------- candado y manifiesto (F15) ---
def estado_git():
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=v.RAIZ, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": git("rev-parse", "HEAD"), "cambios_en_src_o_docs": git("status", "--porcelain", "--", "src", "docs")}


def manifiesto(extra=None):
    import lightgbm, mapie, neuralforecast, statsmodels, torch
    return {"fecha": pd.Timestamp.now().isoformat(timespec="seconds"), "git": estado_git(),
            "huella_dataset": int(v.HUELLA_OFICIAL), "arnes": ARNES,
            "python": sys.version.split()[0], "sistema": platform.platform(),
            "versiones": {"pandas": pd.__version__, "numpy": np.__version__, "statsmodels": statsmodels.__version__,
                          "lightgbm": lightgbm.__version__, "neuralforecast": neuralforecast.__version__,
                          "mapie": mapie.__version__, "torch": torch.__version__},
            **(extra or {})}
