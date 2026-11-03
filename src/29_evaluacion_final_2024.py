"""Lectura única de la evaluación final en 2024 (protocolo 15). SE EJECUTA UNA SOLA VEZ.

Lee las previsiones que dejó 28_prevision_final_2024.py y calcula, en este orden: (1) controles
de lectura (15.7.7 a 15.7.10; se detiene si falla alguno); (2) métricas puntuales y comparación
con 2023 (15.5.1); (3) Diebold-Mariano (15.6); (4) intervalos propios y rejilla de 19 cuantiles;
(5) ventana y ACI con γ = 0,001 sobre la concatenación 2023 + 2024 con el arranque D7 (15.4);
sensibilidad a γ (descriptiva); (6) tablas CSV y figuras en data/resultados/final/; (7) la
anticipación de 15.9 frente a lo observado, comprobación a comprobación.

Candado (F15): si ya existe data/resultados/final/manifiesto.json, el script se niega a
ejecutarse salvo con --repetir-por-error "motivo", que queda anotado en el manifiesto.
Nada de lo que se lea modifica un modelo ni el protocolo (15.1.1).
"""
import argparse
import json
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import finales as F  # noqa: E402
import probabilistica as P  # noqa: E402
import validacion as v  # noqa: E402

capa = F.cargar_script("27_capa_probabilistica.py")      # reutiliza metricas_modelo, diferencia, elegir_gamma
MODELOS, NOMBRE, DIR = F.MODELOS, F.NOMBRE, F.DIR_RESULTADOS
CORTES_CAUSALIDAD = (465, 565, 665)                     # días 100, 200 y 300 de 2024 en la concatenación (15.7.7)
PRIMER_DIA = 365                                        # primer día emitido: 1/1/2024 (D7)
GAMMA = 0.001                                           # fijado en 14.13.4 (F4)
GAMMAS = [0.0, *P.GAMMAS]
PARES = [(a, b) for i, a in enumerate(MODELOS) for b in MODELOS[i + 1:]]


# ----------------------------------------------------------------------------- candado ---
def candado(repetir):
    ruta = DIR / "manifiesto.json"
    DIR.mkdir(parents=True, exist_ok=True)
    manifiesto = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None
    if manifiesto and not repetir:
        raise SystemExit("La evaluación final ya se ejecutó (manifiesto.json). Solo se repite por un error de "
                         "programación: --repetir-por-error \"motivo\" (protocolo 2.6).")
    nuevo = F.manifiesto({"estado": "iniciada"})
    if manifiesto:
        nuevo["repeticiones"] = manifiesto.get("repeticiones", []) + [
            {"motivo": repetir, "ejecucion_anterior": {k: manifiesto[k] for k in ("fecha", "git") if k in manifiesto}}]
    ruta.write_text(json.dumps(nuevo, indent=1, default=str), encoding="utf-8")
    return nuevo


def cerrar_manifiesto(nuevo, extra):
    nuevo.update(estado="completa", **extra)
    (DIR / "manifiesto.json").write_text(json.dumps(nuevo, indent=1, default=str), encoding="utf-8")


# ------------------------------------------------------------------------------- datos ---
def cargar():
    datos = v.cargar_datos(incluir_prueba=True)
    o23, o24 = v.generar_origenes("validacion"), v.generar_origenes("prueba")
    T23 = {m: pd.read_parquet(v.DIR_PREDICCIONES / f"{m}_cuantiles_validacion.parquet") for m in MODELOS}
    T24 = {m: pd.read_parquet(v.DIR_PREDICCIONES / f"{m}_cuantiles_prueba.parquet") for m in MODELOS}
    ref = T24["ingenuo"]
    for m in MODELOS:
        for t, n in ((T23[m], 8760), (T24[m], 8784)):
            assert len(t) == n and not t.isna().any().any(), f"{m}: filas o nulos"
            assert (np.diff(t[P.COLUMNAS_Q].to_numpy(), axis=1) >= 0).all(), f"{m}: rejilla no monótona"
        assert np.array_equal(P.a_ns(T24[m]["hora_utc"]), P.a_ns(ref["hora_utc"])), f"{m}: horas distintas"
        assert np.array_equal(T24[m]["real"].to_numpy(), ref["real"].to_numpy()), f"{m}: `real` distinta"
        assert np.array_equal(P.a_ns(T23[m]["hora_utc"]), P.a_ns(T23["ingenuo"]["hora_utc"]))
    real24 = datos["demanda_mw"].reindex(pd.DatetimeIndex(ref["hora_utc"]).as_unit(datos.index.unit))
    assert np.array_equal(real24.to_numpy(), ref["real"].to_numpy()), "`real` de 2024 distinta del dataset"
    propios = {}
    for m in MODELOS:
        if m in ("lightgbm", "nhits"):
            o = pd.read_parquet(v.DIR_PREDICCIONES / f"{m}_intervalos_prueba.parquet")
            assert np.array_equal(o["prevision"].to_numpy(), T24[m]["prevision"].to_numpy())
            propios[m] = (o["inferior"].to_numpy(float), o["superior"].to_numpy(float))
        else:
            propios[m] = (T24[m]["q05"].to_numpy(float), T24[m]["q95"].to_numpy(float))

    def piezas(t, o):
        antic, bloque = P.anticipacion_y_bloque(t, o)
        fechas = t["fecha_objetivo"]
        return (P.a_ns(t["hora_utc"]), P.a_ns(fechas.map(o.set_index("fecha_objetivo")["origen_utc"])),
                t["real"].to_numpy(float), bloque, pd.factorize(fechas)[0],
                pd.to_datetime(fechas).dt.month.to_numpy())
    h3, or3, y3, b3, d3, m3 = piezas(T23["ingenuo"], o23)
    h4, or4, y4, b4, d4, m4 = piezas(ref, o24)
    assert d3.max() == 364 and d4.max() == 365
    comun = {"hora_utc": np.concatenate([h3, h4]), "origen_utc": np.concatenate([or3, or4]),
             "real": np.concatenate([y3, y4]), "bloque": np.concatenate([b3, b4]),
             "dia": np.concatenate([d3, d4 + PRIMER_DIA]), "mes": np.concatenate([m3, m4])}
    assert (np.diff(comun["hora_utc"].astype("int64")) > 0).all()
    prevision = {m: np.concatenate([T23[m]["prevision"].to_numpy(float), T24[m]["prevision"].to_numpy(float)])
                 for m in MODELOS}
    return datos, T23, T24, propios, comun, prevision, len(y3)


def completar(x24, n23):
    """Lleva un arreglo de 2024 al eje concatenado (2023 + 2024), con NaN en 2023."""
    x24 = np.asarray(x24, float)
    return np.concatenate([np.full((n23,) + x24.shape[1:], np.nan), x24])


# ----------------------------------------------------------------------------- controles ---
def controles(comun, prevision, datos, T24, escala):
    t0 = time.perf_counter()
    r = F.controles_dm()
    print(f"  15.7.8 Diebold-Mariano sintético: nivel {100 * r['nivel']:.1f} %, potencia {100 * r['potencia']:.1f} %, "
          f"AR(1) {100 * r['ar1']:.1f} %")
    assert 0.03 <= r["nivel"] <= 0.08 and r["potencia"] > 0.80 and r["ar1"] < 0.10, "falla el control del contraste"
    iguales = P.control_recursion_mapie()
    print(f"  15.7.7 recursión propia = MAPIE al bit: {iguales}")
    assert all(iguales.values())
    for m in MODELOS:
        e = {k: comun[k] for k in ("hora_utc", "origen_utc", "real", "bloque", "dia")} | {"prevision": prevision[m]}
        res = P.control_causalidad(e, PRIMER_DIA, CORTES_CAUSALIDAD)
        print(f"  15.7.7 causalidad {NOMBRE[m]} (corte, sin cambio ≤ corte, cambia después, la variante con fuga cambia): {res}")
        assert all(a and b and c for _, a, b, c in res), f"falla el corte de causalidad de ACI en {m}"
    ok, salida = P.control_sintetico()
    assert ok, "ACI no cubre 0,90 ± 0,02 en la prueba sintética"
    print("  15.7.7 sintético de ACI superado")
    assert abs(escala - 1328.2) < 0.1, f"escala del MASE {escala:.2f}"
    for m in MODELOS:                                   # 15.7.9: dos implementaciones independientes
        t = T24[m]
        a = v.metricas(t, escala)
        e = t["real"].to_numpy() - t["prevision"].to_numpy()
        b = {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "MASE": np.abs(e).mean() / escala}
        assert all(abs(a[k] - b[k]) <= 1e-9 * max(1, abs(b[k])) for k in b), f"{m}: métricas distintas"
    meta_fugas = sum(F.leer_meta(c, k)["fugas"] for c in F.COMPONENTES for k in range(1, 13))
    assert meta_fugas == 0, "28 dejó pruebas de fuga con días con fuga"
    print(f"  15.7.9 métricas puntuales reproducidas con dos implementaciones; fugas registradas por 28: {meta_fugas}")
    print(f"  controles superados ({time.perf_counter() - t0:.0f} s)", flush=True)


# --------------------------------------------------------------------------------- main ---
def lectura_cobertura(c):
    return "bien calibrada" if 0.88 <= c <= 0.92 else ("infracobertura" if c < 0.88 else "sobrecobertura")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repetir-por-error", default=None, metavar="MOTIVO")
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    manifiesto = candado(a.repetir_por_error)
    datos, T23, T24, propios, comun, prevision, n23 = cargar()
    y, dia, bloque, mes = comun["real"], comun["dia"], comun["bloque"], comun["mes"]
    E = dia >= PRIMER_DIA
    n24 = int(E.sum())
    assert n24 == 8784
    escala = v.escala_mase(datos)
    print(f"Datos: {len(y)} horas (2023 + 2024); evaluación sobre las {n24} horas de 2024. Controles de entrada superados.")
    print("\n1. Controles de lectura (15.7)")
    controles(comun, prevision, datos, T24, escala)

    # ---------------------------------------------------------------- 2. Puntuales ---
    print("\n2. Métricas puntuales de 2024 (error = real − previsto; MW)")
    fil, abs_dia, cuad_dia = [], {}, {}
    y24 = y[E]
    for m in MODELOS:
        t = T24[m]
        e = t["real"].to_numpy() - t["prevision"].to_numpy()
        dias24 = dia[E]
        met = v.metricas(t, escala)
        mae, lo, hi = P.ic95(np.abs(e), dias24)
        mae23 = np.abs(T23[m]["real"] - T23[m]["prevision"]).mean()
        cambio = 100 * (mae / mae23 - 1)
        fil.append({"modelo": m, "MAE": mae, "MAE_ic_inf": lo, "MAE_ic_sup": hi, "RMSE": met["RMSE"], "MAPE": met["MAPE"],
                    "MASE": met["MASE"], "sesgo": e.mean(), "RMSE_sobre_MAE": met["RMSE"] / mae, "MAE_2023": mae23,
                    "cambio_MAE_pct": cambio,
                    "lectura_vs_2023": "similar" if abs(cambio) <= 10 else ("peor" if cambio > 10 else "mejor"),
                    "sesgo_apreciable": abs(e.mean()) > 150})
        d = pd.DataFrame({"f": t["fecha_objetivo"].to_numpy(), "a": np.abs(e), "c": e ** 2})
        g = d.groupby("f")[["a", "c"]].mean().sort_index()
        abs_dia[m], cuad_dia[m] = g["a"].to_numpy(), g["c"].to_numpy()
    pu = pd.DataFrame(fil).set_index("modelo")
    pu.insert(1, "MAE_rel_ingenuo", pu["MAE"] / pu.loc["ingenuo", "MAE"])
    pu.to_csv(DIR / "puntuales.csv")
    print(pu.round(3).T.to_string())
    por = []
    for m in MODELOS:
        t = T24[m]
        e = t["real"].to_numpy() - t["prevision"].to_numpy()
        for k in range(1, 13):
            s = mes[E] == k
            por.append({"modelo": m, "eje": "mes", "valor": k, "MAE": np.abs(e[s]).mean(), "sesgo": e[s].mean()})
        for b in range(P.N_BLOQUES):
            s = bloque[E] == b
            por.append({"modelo": m, "eje": "bloque", "valor": b + 1, "MAE": np.abs(e[s]).mean(), "sesgo": e[s].mean()})
    por = pd.DataFrame(por)
    por.to_csv(DIR / "puntuales_mes_bloque.csv", index=False)
    print("\n  MAE por mes (filas) y modelo:")
    print(por[por["eje"] == "mes"].pivot(index="valor", columns="modelo", values="MAE")[list(MODELOS)].round(0).to_string())
    print("  MAE por bloque (B1 00-05 h locales, B2 06-11, B3 12-17, B4 18-23):")
    print(por[por["eje"] == "bloque"].pivot(index="valor", columns="modelo", values="MAE")[list(MODELOS)].round(0).to_string())

    # --------------------------------------------------------------- 3. Diebold-Mariano ---
    print("\n3. Diebold-Mariano (pérdida diaria, Newey-West con 7 retardos, HLN, t de 365 g. l., Holm)")
    dm = []
    for perdida, dic in (("absoluta", abs_dia), ("cuadratica", cuad_dia)):
        for retardo in (7, 3, 14):
            filas = []
            for aa, bb in PARES:
                d = dic[aa] - dic[bb]
                est, p = F.diebold_mariano(d, retardo)
                filas.append({"a": aa, "b": bb, "perdida": perdida, "retardo": retardo, "diferencia_media_a_menos_b": d.mean(),
                              "DM": est, "p": p})
            ph = F.holm([f["p"] for f in filas])
            for f, x in zip(filas, ph):
                f["p_holm"] = x
                f["lectura"] = "apreciable" if x < 0.05 else "no distinguible"
                f["mejor"] = f["a"] if f["diferencia_media_a_menos_b"] < 0 else f["b"]
            dm += filas
    dm = pd.DataFrame(dm)
    dm.to_csv(DIR / "diebold_mariano.csv", index=False)
    principal = dm[(dm["perdida"] == "absoluta") & (dm["retardo"] == 7)]
    print(principal[["a", "b", "diferencia_media_a_menos_b", "DM", "p", "p_holm", "lectura", "mejor"]].round(4).to_string(index=False))
    print("  Robustez (pérdida cuadrática, retardo 7):")
    print(dm[(dm["perdida"] == "cuadratica") & (dm["retardo"] == 7)][["a", "b", "DM", "p_holm", "lectura"]].round(4).to_string(index=False))
    print("  Sensibilidad al retardo (pérdida absoluta, p ajustado por Holm):")
    print(dm[dm["perdida"] == "absoluta"].pivot_table(index=["a", "b"], columns="retardo", values="p_holm").round(4).to_string())

    # ----------------------------------------------------------- 4. Intervalos propios ---
    print("\n4. Intervalos propios del 90 % y rejilla de 19 cuantiles (8.784 horas)")
    horas_is, horas_crps, filas = {}, {}, []
    Qp = {m: completar(T24[m][P.COLUMNAS_Q].to_numpy(float), n23) for m in MODELOS}
    lo_p = {m: completar(propios[m][0], n23) for m in MODELOS}
    hi_p = {m: completar(propios[m][1], n23) for m in MODELOS}
    for m in MODELOS:
        horas_is[m] = P.puntuacion_intervalo(y, lo_p[m], hi_p[m])
        horas_crps[m] = P.crps19(y, Qp[m])
        mm = capa.metricas_modelo(y, lo_p[m], hi_p[m], Qp[m], E, comun)
        mm["cobertura_lectura"] = lectura_cobertura(mm["cobertura"])
        filas.append({"modelo": m, **mm,
                      "MAE_prevision": np.abs(y - completar(T24[m]["prevision"], n23))[E].mean()})
    propios_t = pd.DataFrame(filas).set_index("modelo")
    propios_t.to_csv(DIR / "propios_metricas.csv")
    cols = ["cobertura", "cob_ic_inf", "cob_ic_sup", "cobertura_lectura", "fallo_debajo", "fallo_encima", "cociente_fallos",
            "ancho_medio", "ancho_mediana", "IS", "IS_ic_inf", "IS_ic_sup", "pinball05", "pinball95", "CRPS19",
            "CRPS19_ic_inf", "CRPS19_ic_sup", "fiab_desv_max", "MAE_prevision"]
    print(propios_t[cols].T.to_string(float_format=lambda x: f"{x:.4g}"))
    print("  Cobertura por bloque (B1 00-05 h, B2 06-11, B3 12-17, B4 18-23):")
    print(propios_t[[f"cob_B{b}" for b in range(1, 5)]].round(3).to_string())
    print("  Cobertura por mes:")
    print(propios_t[[f"cob_mes{k:02d}" for k in range(1, 13)]].round(2).to_string())
    fiab_p = pd.DataFrame({m: P.fiabilidad(y[E], Qp[m][E]) for m in MODELOS}, index=P.NIVELES)
    fiab_p.index.name = "tau"
    fiab_p.to_csv(DIR / "fiabilidad_propios.csv")
    print("  Fiabilidad de la rejilla propia (frecuencia observada de real ≤ q_τ):")
    print(fiab_p.round(3).T.to_string())
    pares = []
    for aa, bb in PARES:
        for nombre, h in (("IS", horas_is), ("CRPS19", horas_crps)):
            media, lo_, hi_, lect = capa.diferencia(h[aa][E], h[bb][E], dia[E])
            pares.append({"a": aa, "b": bb, "metrica": nombre, "diferencia_a_menos_b": media, "ic_inf": lo_,
                          "ic_sup": hi_, "lectura": lect})
    pares = pd.DataFrame(pares)
    pares.to_csv(DIR / "propios_pares.csv", index=False)
    print("  Diferencia pareada a − b (negativa = a mejor), IC del 95 % por bootstrap de días:")
    print(pares.round(1).to_string(index=False))

    # ----------------------------------------------------------------- 5. Ventana y ACI ---
    print("\n5. Ventana (γ = 0) y ACI (γ = 0,001) con arranque D7; sensibilidad a γ")
    configs = [(P.ALFA, g) for g in GAMMAS]
    resultados, tabla_is, filas_g = {}, {}, []
    for m in MODELOS:
        t0 = time.perf_counter()
        r = P.motor_aci(comun["hora_utc"], comun["origen_utc"], y, prevision[m], bloque, dia, configs, PRIMER_DIA)
        resultados[m], tabla_is[m] = r, {}
        for c, g in enumerate(GAMMAS):
            lo, hi = r["inferior"][c], r["superior"][c]
            mm = capa.metricas_modelo(y, lo, hi, None, E, comun)
            tabla_is[m][g] = mm["IS"]
            filas_g.append({"modelo": m, "gamma": g, "cobertura": mm["cobertura"], "ancho_medio": mm["ancho_medio"],
                            "IS": mm["IS"], "horas_infinitas": int(r["horas_infinitas"][c]),
                            "alfa_t_medio": float(np.nanmean(r["alfa_emitido"][c][E]))})
        print(f"  {NOMBRE[m]}: ACI en {len(configs)} configuraciones ({time.perf_counter() - t0:.0f} s)", flush=True)
    tg = pd.DataFrame(filas_g)
    tg["IS_rel_g0"] = tg["IS"] / tg.groupby("modelo")["IS"].transform(lambda s: s.iloc[0])
    tg.to_csv(DIR / "aci_gamma.csv", index=False)
    print(tg.round({"cobertura": 3, "ancho_medio": 0, "IS": 0, "IS_rel_g0": 4, "alfa_t_medio": 3}).to_string(index=False))
    _, puntuacion = capa.elegir_gamma(tabla_is)
    print("  Puntuación de la regla 14.6.7 recalculada con 2024 (solo descriptiva): "
          + ", ".join(f"γ {g}: {p:.4f}" for g, p in puntuacion.items()))
    c_aci = GAMMAS.index(GAMMA)
    filas_v, filas_a, QV, QA, hIS_v, hIS_a, hC_v, hC_a = [], [], {}, {}, {}, {}, {}, {}
    for m in MODELOS:
        for gamma, destino, Qd, hIS, hC, filas_d in ((0.0, "ventana", QV, hIS_v, hC_v, filas_v),
                                                    (GAMMA, "ACI", QA, hIS_a, hC_a, filas_a)):
            Q, reord, inf = P.cuantiles_de_aci(comun["hora_utc"], comun["origen_utc"], y, prevision[m], bloque, dia,
                                               PRIMER_DIA, gamma)
            Qd[m] = Q
            c = GAMMAS.index(gamma)
            lo, hi = resultados[m]["inferior"][c], resultados[m]["superior"][c]
            hIS[m] = P.puntuacion_intervalo(y, lo, hi)
            hC[m] = P.crps19(y, np.where(np.isnan(Q), 0.0, Q))
            mm = capa.metricas_modelo(y, lo, hi, Q, E, comun)
            inf90 = int(resultados[m]["horas_infinitas"][c])
            filas_d.append({"modelo": m, "gamma": gamma, **mm, "cobertura_lectura": lectura_cobertura(mm["cobertura"]),
                            "horas_infinitas_90": inf90, "pct_infinitas_90": 100 * inf90 / n24,
                            "horas_reordenadas": reord, "horas_infinitas_niveles": int(inf.sum())})
        pd.DataFrame({"hora_utc": T24[m]["hora_utc"], "fecha_objetivo": T24[m]["fecha_objetivo"], "real": T24[m]["real"],
                      "prevision": T24[m]["prevision"],
                      **{f"inf_g{g}": resultados[m]["inferior"][i][E] for i, g in enumerate(GAMMAS)},
                      **{f"sup_g{g}": resultados[m]["superior"][i][E] for i, g in enumerate(GAMMAS)},
                      "alfa_t": resultados[m]["alfa_emitido"][c_aci][E],
                      **{f"aci_{col}": QA[m][E][:, j] for j, col in enumerate(P.COLUMNAS_Q)}}
                     ).to_parquet(v.DIR_PREDICCIONES / f"{m}_aci_prueba.parquet", index=False)
    ven_t, aci_t = pd.DataFrame(filas_v).set_index("modelo"), pd.DataFrame(filas_a).set_index("modelo")
    ven_t.to_csv(DIR / "ventana_metricas.csv")
    aci_t.to_csv(DIR / "aci_metricas.csv")
    cols = ["cobertura", "cob_ic_inf", "cob_ic_sup", "cobertura_lectura", "fallo_debajo", "fallo_encima", "cociente_fallos",
            "ancho_medio", "ancho_mediana", "IS", "IS_ic_inf", "IS_ic_sup", "CRPS19", "CRPS19_ic_inf", "CRPS19_ic_sup",
            "fiab_desv_max", "horas_infinitas_90", "pct_infinitas_90", "horas_reordenadas"]
    for nombre, tabla in (("Ventana (γ = 0)", ven_t), (f"ACI (γ = {GAMMA})", aci_t)):
        print(f"\n  {nombre}:")
        print(tabla[cols].T.to_string(float_format=lambda x: f"{x:.4g}"))
        print(tabla[[f"cob_B{b}" for b in range(1, 5)]].round(3).to_string())
        print(tabla[[f"cob_mes{k:02d}" for k in range(1, 13)]].round(2).to_string())
    comp = []
    for m in MODELOS:
        for ref_n, h_is, h_c in (("propio", horas_is, horas_crps), ("ventana", hIS_v, hC_v)):
            for nombre, ha, hp in (("IS", hIS_a, h_is), ("CRPS19", hC_a, h_c)):
                media, lo_, hi_, lect = capa.diferencia(ha[m][E], hp[m][E], dia[E])
                comp.append({"modelo": m, "metrica": nombre, "referencia": ref_n, "aci_menos_referencia": media,
                             "ic_inf": lo_, "ic_sup": hi_, "lectura": lect, "referencia_valor": hp[m][E].mean(),
                             "aci": ha[m][E].mean()})
    comp = pd.DataFrame(comp)
    comp.to_csv(DIR / "aci_comparaciones.csv", index=False)
    print("\n  ACI (γ = 0,001) frente al intervalo propio y frente a la ventana (negativa = ACI mejor):")
    print(comp.round({"aci_menos_referencia": 1, "ic_inf": 1, "ic_sup": 1, "referencia_valor": 1, "aci": 1}).to_string(index=False))
    fiab_a = pd.DataFrame({m: P.fiabilidad(y[E], QA[m][E]) for m in MODELOS}, index=P.NIVELES)
    fiab_v = pd.DataFrame({m: P.fiabilidad(y[E], QV[m][E]) for m in MODELOS}, index=P.NIVELES)
    for nom, f in (("aci", fiab_a), ("ventana", fiab_v)):
        f.index.name = "tau"
        f.to_csv(DIR / f"fiabilidad_{nom}.csv")
    print("  Fiabilidad de la rejilla con ACI:")
    print(fiab_a.round(3).T.to_string())

    # Lectura de 15.8.6: ¿se repite la infracobertura de 2023?
    for m in ("lightgbm", "nhits"):
        r = propios_t.loc[m]
        se_repite = r["cobertura"] < 0.88 and r["cob_ic_sup"] < 0.90
        print(f"  15.8.6 {NOMBRE[m]}: cobertura propia {100 * r['cobertura']:.1f} % "
              f"(IC {100 * r['cob_ic_inf']:.1f}-{100 * r['cob_ic_sup']:.1f}); la infracobertura de 2023 "
              f"{'se repite' if se_repite else 'NO se repite'}")

    figuras(T24, propios, comun, resultados, c_aci, E, fiab_p, fiab_a, propios_t, n23, prevision)
    anticipacion(pu, por, dm, propios_t, ven_t, aci_t, comp, tg, puntuacion, fiab_a)
    cerrar_manifiesto(manifiesto, {"puntuacion_14_6_7": {str(k): x for k, x in puntuacion.items()}})
    print("\nListo. Tablas y figuras en", DIR.relative_to(v.RAIZ))


# ------------------------------------------------------------------------------ figuras ---
def figuras(T24, propios, comun, resultados, c_aci, E, fiab_p, fiab_a, propios_t, n23, prevision):
    y = comun["real"][E]
    fig, ejes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)
    for ax, m in zip(ejes, MODELOS):
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="calibración perfecta")
        ax.plot(P.NIVELES, fiab_p[m], marker="o", ms=3, label="propia")
        ax.plot(P.NIVELES, fiab_a[m], marker="s", ms=3, label="con ACI")
        ax.set_title(NOMBRE[m]); ax.set_xlabel("nivel τ del cuantil"); ax.grid(alpha=0.3)
    ejes[0].set_ylabel("frecuencia observada de real ≤ q_τ"); ejes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(DIR / "fiabilidad.png", dpi=130); plt.close(fig)

    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.2))
    for m in MODELOS:
        ejes[0].plot(range(1, 5), [propios_t.loc[m, f"cob_B{b}"] for b in range(1, 5)], marker="o", label=NOMBRE[m])
        ejes[1].plot(range(1, 13), [propios_t.loc[m, f"cob_mes{k:02d}"] for k in range(1, 13)], marker="o", label=NOMBRE[m])
    for ax, titulo, et in zip(ejes, ("por bloque de anticipación", "por mes"), ("bloque (B1 00-05 h … B4 18-23 h locales)", "mes de 2024")):
        ax.axhline(0.9, color="0.4", ls="--", lw=1); ax.set_title(f"Cobertura propia del 90 %, {titulo}")
        ax.set_xlabel(et); ax.set_ylabel("cobertura"); ax.grid(alpha=0.3)
    ejes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(DIR / "cobertura_bloque_mes.png", dpi=130); plt.close(fig)

    fechas = pd.to_datetime(T24["ingenuo"]["fecha_objetivo"])
    sel = ((fechas >= F.SEMANA[0]) & (fechas <= F.SEMANA[1])).to_numpy()
    x = np.arange(sel.sum())
    fig, ejes = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey=True)
    for ax, m in zip(ejes.ravel(), MODELOS):
        lo, hi = propios[m]
        ax.fill_between(x, lo[sel], hi[sel], alpha=0.25, label="propio 90 %")
        ax.fill_between(x, resultados[m]["inferior"][c_aci][E][sel], resultados[m]["superior"][c_aci][E][sel],
                        alpha=0.25, color="tab:orange", label="ACI 90 %")
        ax.plot(x, T24[m]["prevision"].to_numpy()[sel], lw=1, color="k", label="previsión")
        ax.plot(x, y[sel], lw=1, color="tab:red", label="real"); ax.set_title(NOMBRE[m]); ax.grid(alpha=0.3)
    ejes[0, 0].legend(fontsize=8); fig.supxlabel(f"horas del {F.SEMANA[0]} al {F.SEMANA[1]} (días objetivo)")
    fig.supylabel("MW"); fig.tight_layout(); fig.savefig(DIR / "semana_ejemplo.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------- anticipación ---
def anticipacion(pu, por, dm, pr, ven_t, aci_t, comp, tg, puntuacion, fiab_a):
    """Tabla automática de 15.9: lo anticipado frente a lo observado (✓ dentro; ✗ FALLA si no)."""
    filas = []

    def rango(i, nombre, valor, bajo, alto, unidad=""):
        filas.append((i, nombre, f"{bajo:g} a {alto:g}{unidad}", f"{valor:.4g}{unidad}",
                      "✓" if bajo <= valor <= alto else "✗ FALLA"))

    def cond(i, nombre, esperado, observado, ok):
        filas.append((i, nombre, esperado, str(observado), "✓" if ok else "✗ FALLA"))

    N = NOMBRE
    cond("A0", "Todos los controles de 15.7 pasan", "sí", "sí (el script no habría llegado aquí)", True)
    for i, (m, (b, a_)) in zip(("A1", "A2", "A3", "A4"), {"ingenuo": (1220, 1500), "sarimax": (830, 1060),
                                                         "lightgbm": (540, 680), "nhits": (650, 850)}.items()):
        rango(i, f"MAE {N[m]}", pu.loc[m, "MAE"], b, a_, " MW")
    orden = list(pu["MAE"].sort_values().index)
    esperado = ["lightgbm", "nhits", "sarimax", "ingenuo"]
    cond("A5", "Orden por MAE (mejor→peor)", " < ".join(esperado), " < ".join(orden), orden == esperado)
    cond("A6", "LightGBM es el de menor MAE", "sí", orden[0], orden[0] == "lightgbm")
    cond("A7", "N-HiTS mejor que SARIMAX (MAE)", "sí", f"{pu.loc['nhits', 'MAE']:.0f} frente a {pu.loc['sarimax', 'MAE']:.0f}",
         pu.loc["nhits", "MAE"] < pu.loc["sarimax", "MAE"])
    for m, (b, a_) in {"lightgbm": (0.38, 0.52), "nhits": (0.46, 0.64), "sarimax": (0.60, 0.80)}.items():
        rango("A8", f"MAE relativo al ingenuo, {N[m]}", pu.loc[m, "MAE_rel_ingenuo"], b, a_)
    for m, (b, a_) in {"ingenuo": (-100, 180), "sarimax": (-330, 100), "lightgbm": (-250, 60), "nhits": (20, 330)}.items():
        rango("A9", f"Sesgo {N[m]}", pu.loc[m, "sesgo"], b, a_, " MW")
    for m, signo in (("nhits", 1), ("lightgbm", -1), ("sarimax", -1)):
        cond("A10", f"Signo del sesgo, {N[m]}", "> 0" if signo > 0 else "< 0", f"{pu.loc[m, 'sesgo']:+.0f}",
             signo * pu.loc[m, "sesgo"] > 0)
    r = pu["RMSE_sobre_MAE"]
    cond("A11", "RMSE/MAE entre 1,18 y 1,62 en los cuatro", "sí", ", ".join(f"{x:.2f}" for x in r), bool(((r >= 1.18) & (r <= 1.62)).all()))
    cond("A11", "RMSE/MAE de N-HiTS menor que el de SARIMAX", "sí", f"{r['nhits']:.2f} frente a {r['sarimax']:.2f}", r["nhits"] < r["sarimax"])
    for m in MODELOS:
        cond("A12", f"Cambio del MAE frente a 2023 dentro de ±10 %, {N[m]}", "similar",
             f"{pu.loc[m, 'cambio_MAE_pct']:+.1f} % ({pu.loc[m, 'lectura_vs_2023']})", abs(pu.loc[m, "cambio_MAE_pct"]) <= 10)
    cond("A13", "LightGBM y N-HiTS no mejoran su MAE de 2023", "cambio ≥ 0 en ambos",
         f"{pu.loc['lightgbm', 'cambio_MAE_pct']:+.1f} % y {pu.loc['nhits', 'cambio_MAE_pct']:+.1f} %",
         bool((pu.loc[["lightgbm", "nhits"], "cambio_MAE_pct"] >= 0).all()))
    cond("A14", "|Sesgo| del SARIMAX ≤ 196 MW (el reajuste lo reduce)", "sí", f"{pu.loc['sarimax', 'sesgo']:+.0f} MW",
         abs(pu.loc["sarimax", "sesgo"]) <= 196)
    prin = dm[(dm["perdida"] == "absoluta") & (dm["retardo"] == 7)].set_index(["a", "b"])
    cuad = dm[(dm["perdida"] == "cuadratica") & (dm["retardo"] == 7)].set_index(["a", "b"])

    def par(x, y_, tabla):
        return tabla.loc[(x, y_)] if (x, y_) in tabla.index else tabla.loc[(y_, x)]
    esperados = {("lightgbm", "sarimax"): "A15", ("lightgbm", "nhits"): "A15", ("nhits", "sarimax"): "A15",
                 ("sarimax", "ingenuo"): "A15", ("nhits", "ingenuo"): "A15", ("lightgbm", "ingenuo"): "A15"}
    todos = True
    for (x, y_) in esperados:
        r_ = par(x, y_, prin)
        ok = r_["lectura"] == "apreciable" and r_["mejor"] == x
        todos &= ok
        cond("A15", f"{N[x]} apreciablemente mejor que {N[y_]} (Holm, pérdida absoluta)", "sí",
             f"{r_['lectura']}, mejor {N[r_['mejor']]}, p ajustado {r_['p_holm']:.4g}", ok)
    cond("A15", "Los seis pares a la vez", "sí", "sí" if todos else "no", todos)
    iguales = all(par(x, y_, prin)["lectura"] == par(x, y_, cuad)["lectura"] for (x, y_) in esperados)
    cond("A16", "La pérdida cuadrática da la misma conclusión en los seis pares", "sí", "sí" if iguales else "no", iguales)
    for m, (b, a_) in {"ingenuo": (86, 91), "sarimax": (89, 95), "lightgbm": (60, 78), "nhits": (58, 76)}.items():
        rango("A17", f"Cobertura propia, {N[m]}", 100 * pr.loc[m, "cobertura"], b, a_, " %")
    cond("A18", "LightGBM y N-HiTS infracubren (< 88 %)", "sí", f"{100 * pr.loc['lightgbm', 'cobertura']:.1f} y {100 * pr.loc['nhits', 'cobertura']:.1f} %",
         bool((pr.loc[["lightgbm", "nhits"], "cobertura"] < 0.88).all()))
    peor = max(pr.loc["lightgbm", "cobertura"], pr.loc["nhits", "cobertura"])
    cond("A18", "Los dos por debajo del ingenuo y del SARIMAX", "sí", f"máx. {100 * peor:.1f} % frente a mín. {100 * pr.loc[['ingenuo', 'sarimax'], 'cobertura'].min():.1f} %",
         peor < pr.loc[["ingenuo", "sarimax"], "cobertura"].min())
    cf = pr["cociente_fallos"]
    cond("A19", "Cociente de fallos del SARIMAX > 1,3", "> 1,3", f"{cf['sarimax']:.2f}", cf["sarimax"] > 1.3)
    cond("A19", "Cociente de fallos de LightGBM > 1,2", "> 1,2", f"{cf['lightgbm']:.2f}", cf["lightgbm"] > 1.2)
    cond("A19", "Cociente de fallos de N-HiTS < 0,9", "< 0,9", f"{cf['nhits']:.2f}", cf["nhits"] < 0.9)
    cond("A19", "Cociente de fallos del ingenuo entre 0,7 y 1,4", "0,7 a 1,4", f"{cf['ingenuo']:.2f}", 0.7 <= cf["ingenuo"] <= 1.4)
    for m, (b, a_) in {"ingenuo": (5500, 6300), "sarimax": (4100, 4800), "lightgbm": (1600, 2400), "nhits": (1500, 2300)}.items():
        rango("A20", f"Anchura media propia, {N[m]}", pr.loc[m, "ancho_medio"], b, a_, " MW")
    for m, (b, a_) in {"ingenuo": (8800, 10200), "sarimax": (6100, 7300), "lightgbm": (4400, 5500), "nhits": (4500, 5600)}.items():
        rango("A21", f"IS propio, {N[m]}", pr.loc[m, "IS"], b, a_, " MW")
    o_is, o_c = list(pr["IS"].sort_values().index), list(pr["CRPS19"].sort_values().index)
    cond("A22", "Orden por IS propio (mejor→peor)", " < ".join(esperado), " < ".join(o_is), o_is == esperado)
    for m, (b, a_) in {"ingenuo": (950, 1120), "sarimax": (640, 780), "lightgbm": (400, 500), "nhits": (470, 600)}.items():
        rango("A23", f"CRPS₁₉ propio, {N[m]}", pr.loc[m, "CRPS19"], b, a_, " MW")
    cond("A24", "Orden por CRPS₁₉ propio (mejor→peor)", " < ".join(esperado), " < ".join(o_c), o_c == esperado)
    rr = pr["CRPS19"] / pr["MAE_prevision"]
    cond("A25", "CRPS₁₉/MAE entre 0,70 y 0,80 en los cuatro", "sí", ", ".join(f"{x:.3f}" for x in rr), bool(((rr >= 0.70) & (rr <= 0.80)).all()))
    fd = pr["fiab_desv_max"]
    cond("A26", "Desviación máx. de fiabilidad > 0,06 en SARIMAX, LightGBM y N-HiTS", "sí",
         ", ".join(f"{N[m]} {fd[m]:.3f}" for m in ("sarimax", "lightgbm", "nhits")), bool((fd[["sarimax", "lightgbm", "nhits"]] > 0.06).all()))
    cond("A26", "Desviación máx. de fiabilidad del ingenuo ≤ 0,06", "sí", f"{fd['ingenuo']:.3f}", fd["ingenuo"] <= 0.06)
    cond("A27", "Cobertura propia del SARIMAX en B1 mayor que en B3 en más de 5 puntos", "sí",
         f"B1 {100 * pr.loc['sarimax', 'cob_B1']:.1f} % y B3 {100 * pr.loc['sarimax', 'cob_B3']:.1f} %",
         pr.loc["sarimax", "cob_B1"] - pr.loc["sarimax", "cob_B3"] > 0.05)
    ca = aci_t["cobertura"]
    cond("A28", "Cobertura con ACI entre 88 y 92 % en los cuatro", "sí", ", ".join(f"{100 * x:.1f}" for x in ca), bool(((ca >= 0.88) & (ca <= 0.92)).all()))
    cond("A28", "Cobertura con ACI entre 86 y 93 % en los cuatro", "sí", ", ".join(f"{100 * x:.1f}" for x in ca), bool(((ca >= 0.86) & (ca <= 0.93)).all()))
    ci = comp[(comp["referencia"] == "propio") & (comp["metrica"] == "IS")].set_index("modelo")
    for m in ("lightgbm", "nhits"):
        rel = 100 * ci.loc[m, "aci_menos_referencia"] / ci.loc[m, "referencia_valor"]
        cond("A29", f"IS de ACI frente al propio, {N[m]}: baja entre 15 y 40 % y apreciable", "−40 a −15 % y apreciable",
             f"{rel:+.1f} % ({ci.loc[m, 'lectura']})", -40 <= rel <= -15 and ci.loc[m, "lectura"] == "apreciable")
    cond("A30", "IS de ACI frente al propio, SARIMAX: no distinguible", "no distinguible", ci.loc["sarimax", "lectura"],
         ci.loc["sarimax", "lectura"] == "no distinguible")
    cond("A30", "IS de ACI frente al propio, ingenuo: peor y apreciable", "peor y apreciable",
         f"{ci.loc['ingenuo', 'aci_menos_referencia']:+.0f} MW ({ci.loc['ingenuo', 'lectura']})",
         ci.loc["ingenuo", "aci_menos_referencia"] > 0 and ci.loc["ingenuo", "lectura"] == "apreciable")
    rel0 = tg[tg["gamma"] == GAMMA].set_index("modelo")["IS_rel_g0"]
    cond("A31", "IS(γ = 0,001)/IS(γ = 0) entre 0,99 y 1,02 en los cuatro", "sí", ", ".join(f"{x:.4f}" for x in rel0),
         bool(((rel0 >= 0.99) & (rel0 <= 1.02)).all()))
    cond("A32", "Puntuación de 14.6.7 > 1 para los cuatro γ", "sí", ", ".join(f"{x:.4f}" for x in puntuacion.values()),
         all(x > 1 for x in puntuacion.values()))
    cond("A32", "γ = 0,001 tiene la menor puntuación", "sí", min(puntuacion, key=puntuacion.get), min(puntuacion, key=puntuacion.get) == 0.001)
    i1 = tg[tg["gamma"] == GAMMA].set_index("modelo")["horas_infinitas"]
    i2 = tg[tg["gamma"] == 0.02].set_index("modelo")["horas_infinitas"]
    cond("A33", "Horas infinitas con γ = 0,001: 0 en los cuatro", "sí", ", ".join(str(int(x)) for x in i1), bool((i1 == 0).all()))
    cond("A33", "Horas infinitas con γ = 0,02: > 0 en al menos tres modelos", "sí", ", ".join(str(int(x)) for x in i2), int((i2 > 0).sum()) >= 3)
    fa = aci_t["fiab_desv_max"]
    cond("A34", "Desviación máx. de fiabilidad con ACI ≤ 0,06 en los cuatro", "sí", ", ".join(f"{x:.3f}" for x in fa), bool((fa <= 0.06).all()))
    cc = comp[(comp["referencia"] == "propio") & (comp["metrica"] == "CRPS19")].set_index("modelo")
    ok35 = all(cc.loc[m, "aci_menos_referencia"] < 0 and cc.loc[m, "lectura"] == "apreciable" for m in ("lightgbm", "nhits"))
    cond("A35", "CRPS₁₉ de ACI mejora al propio en LightGBM y N-HiTS, apreciable", "sí",
         ", ".join(f"{N[m]} {cc.loc[m, 'aci_menos_referencia']:+.0f} MW ({cc.loc[m, 'lectura']})" for m in ("lightgbm", "nhits")), ok35)
    meta = {(c, k): F.leer_meta(c, k) for c in F.COMPONENTES for k in range(1, 13)}
    tot = {c: sum(meta[(c, k)]["minutos"] for k in range(1, 13)) for c in F.COMPONENTES}
    horas = sum(tot.values()) / 60
    cond("A36", "Tiempo total de 28 en serie entre 3,5 y 6,5 h", "3,5 a 6,5 h", f"{horas:.2f} h", 3.5 <= horas <= 6.5)
    cond("A36", "N-HiTS > 60 % del total", "sí", f"{100 * (tot['nhits'] + tot['nhits19']) / sum(tot.values()):.0f} %",
         (tot["nhits"] + tot["nhits19"]) / sum(tot.values()) > 0.60)
    rec = max(meta[("nhits19", k)].get("recorte", 0) for k in range(1, 13)), max(meta[("lightgbm", k)].get("recorte", 0) for k in range(1, 13))
    cond("A36", "Ningún recorte de 15.10", "ninguno", f"recorte N-HiTS {rec[0]}, LightGBM {rec[1]}", max(rec) == 0)
    tope = {c: sum(meta[(c, k)]["pasos"] >= 1000 for k in range(1, 13)) for c in ("nhits", "nhits19")}
    cond("A37", "Pasos en el tope de 1.000 en al menos 8 de los 12 reajustes (las dos redes)", "≥ 8 en ambas",
         f"oficial {tope['nhits']}, segunda {tope['nhits19']}", min(tope.values()) >= 8)
    res = pd.DataFrame(filas, columns=["id", "comprobación (15.9)", "anticipado", "observado", "resultado"])
    res.to_csv(DIR / "anticipacion_vs_resultado.csv", index=False)
    print("\n6. Anticipación de 15.9 frente a lo observado")
    print(res.to_string(index=False))
    print(f"  Fallan {int(res['resultado'].str.contains('FALLA').sum())} de {len(res)} comprobaciones automáticas.")


if __name__ == "__main__":
    main()
