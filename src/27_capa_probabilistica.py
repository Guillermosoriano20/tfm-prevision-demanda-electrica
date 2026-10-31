"""Capa probabilística de 2023: calibración de los intervalos propios y ACI (protocolo 14).

Lee las rejillas de 26_cuantiles_validacion.py. Orden: (1) controles de 14.7 (se detiene si
falla alguno); (2) métricas de los intervalos propios del 90 % y de la rejilla (14.5);
(3) ACI con MAPIE, elección de γ con la regla de 14.6.7 y métricas después de ACI;
(4) tablas CSV y figuras en data/resultados/probabilistica/; (5) comparación automática con
la anticipación fijada en 14.9. Solo 2023: no lee 2024. Nada decide sobre los modelos.
"""
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import probabilistica as P  # noqa: E402
import validacion as v  # noqa: E402

MODELOS = ("ingenuo", "sarimax", "lightgbm", "nhits")
NOMBRE = {"ingenuo": "Ingenuo", "sarimax": "SARIMAX", "lightgbm": "LightGBM", "nhits": "N-HiTS"}
DIR = v.RAIZ / "data" / "resultados" / "probabilistica"
CORTES_CAUSALIDAD = (100, 200, 300)                      # días de corte de 14.7.3
SEMANA = ("2023-05-15", "2023-05-21")                    # semana de la figura, fijada de antemano
IS_GAMMA0 = "IS_g0"


def cargar():
    """Rejillas, intervalos propios y arreglos comunes, con los controles de 14.7.1."""
    datos = v.cargar_datos()
    origenes = v.generar_origenes("validacion")
    T = {m: pd.read_parquet(v.DIR_PREDICCIONES / f"{m}_cuantiles_validacion.parquet") for m in MODELOS}
    ref = T["ingenuo"]
    for m, t in T.items():
        assert len(t) == 8760 and not t.isna().any().any(), f"{m}: filas o nulos"
        assert np.array_equal(P.a_ns(t["hora_utc"]), P.a_ns(ref["hora_utc"])), f"{m}: horas distintas"
        assert np.array_equal(t["real"].to_numpy(), ref["real"].to_numpy()), f"{m}: `real` distinta"
        assert (np.diff(t[P.COLUMNAS_Q].to_numpy(), axis=1) >= 0).all(), f"{m}: rejilla no monótona"
    real = datos["demanda_mw"].reindex(pd.DatetimeIndex(ref["hora_utc"]).as_unit(datos.index.unit))
    assert np.array_equal(real.to_numpy(), ref["real"].to_numpy()), "`real` distinta del dataset"
    antic, bloque = P.anticipacion_y_bloque(ref, origenes)
    fechas = ref["fecha_objetivo"]
    comun = {
        "hora_utc": P.a_ns(ref["hora_utc"]),
        "origen_utc": P.a_ns(fechas.map(origenes.set_index("fecha_objetivo")["origen_utc"])),
        "real": ref["real"].to_numpy(float), "bloque": bloque,
        "dia": pd.factorize(fechas)[0], "mes": pd.to_datetime(fechas).dt.month.to_numpy(),
    }
    assert comun["dia"].max() == 364
    propios = {}
    for m in MODELOS:
        if m in ("lightgbm", "nhits"):                      # intervalos oficiales (11.2, 12.2)
            o = pd.read_parquet(v.DIR_PREDICCIONES / f"{m}_intervalos_validacion.parquet")
            assert np.array_equal(P.a_ns(o["hora_utc"]), comun["hora_utc"])
            assert np.array_equal(o["prevision"].to_numpy(), T[m]["prevision"].to_numpy())
            propios[m] = (o["inferior"].to_numpy(float), o["superior"].to_numpy(float))
        else:                                               # cuantiles 0,05 y 0,95 de la rejilla
            propios[m] = (T[m]["q05"].to_numpy(float), T[m]["q95"].to_numpy(float))
    return T, propios, comun


def entradas_de(m, T, comun):
    return {**{k: comun[k] for k in ("hora_utc", "origen_utc", "real", "bloque", "dia")},
            "prevision": T[m]["prevision"].to_numpy(float)}


def controles(T, comun):
    """14.7.2 a 14.7.4. Se detiene si falla alguno."""
    inicio = time.perf_counter()
    iguales = P.control_recursion_mapie()
    print(f"  14.7.2 recursión propia = MAPIE al bit: {iguales}")
    assert all(iguales.values()), "la recursión de ACI no coincide con MAPIE"
    for m in MODELOS:
        r = P.control_causalidad(entradas_de(m, T, comun), P.DIAS_ARRANQUE_2023, CORTES_CAUSALIDAD)
        print(f"  14.7.3 causalidad {NOMBRE[m]} (corte, sin cambio ≤ corte, cambia después, "
              f"la variante con fuga cambia): {r}")
        assert all(a and b and c for _, a, b, c in r), f"falla el corte de causalidad de ACI en {m}"
    ok, salida = P.control_sintetico()
    print("  14.7.4 sintético (cobertura global sin salto / en la cola tras el salto): "
          + ", ".join(f"γ {g}: {salida[('sin_salto', g)][0]:.3f} / {salida[('salto', g)][1]:.3f}" for g in P.GAMMAS))
    assert ok, "ACI no cubre 0,90 ± 0,02 en la prueba sintética"
    print(f"  controles superados ({time.perf_counter() - inicio:.0f} s)", flush=True)


def metricas_modelo(y, lo, hi, Q, mask, comun):
    """Todas las métricas de un intervalo del 90 % y de su rejilla sobre las horas de `mask`."""
    dia, bloque, mes = comun["dia"][mask], comun["bloque"][mask], comun["mes"][mask]
    m = P.metricas_intervalo(y[mask], lo[mask], hi[mask], dia, bloque, mes)
    is_h = P.puntuacion_intervalo(y[mask], lo[mask], hi[mask])
    m["IS_ic_inf"], m["IS_ic_sup"] = P.ic95(is_h, dia)[1:]
    if Q is not None:
        crps = P.crps19(y[mask], Q[mask])
        m["CRPS19"] = crps.mean()
        m["CRPS19_ic_inf"], m["CRPS19_ic_sup"] = P.ic95(crps, dia)[1:]
        fi = P.fiabilidad(y[mask], Q[mask])
        m["fiab_desv_max"] = float(np.abs(fi - P.NIVELES).max())
    return m


def diferencia(a, b, dia):
    """Media e IC del 95 % de la diferencia pareada a − b (bootstrap de días) y su lectura."""
    media, lo, hi = P.ic95(a - b, dia)
    return media, lo, hi, "apreciable" if (lo > 0 or hi < 0) else "no distinguible"


def elegir_gamma(tabla_is):
    """Regla de 14.6.7: menor media de IS_m(γ)/IS_m(0); empates (<0,5 %) → el menor γ."""
    puntuacion = {g: float(np.mean([tabla_is[m][g] / tabla_is[m][0.0] for m in MODELOS])) for g in P.GAMMAS}
    mejor = min(puntuacion.values())
    elegido = min(g for g, p in puntuacion.items() if p <= mejor * 1.005)
    return elegido, puntuacion


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    T, propios, comun = cargar()
    y, dia, bloque, mes = comun["real"], comun["dia"], comun["bloque"], comun["mes"]
    E = dia >= P.DIAS_ARRANQUE_2023
    TODAS = np.ones(len(y), bool)
    print(f"Datos: {len(y)} horas, {int(E.sum())} en E (27/02-31/12). Controles de entrada superados.")
    print("\n1. Controles de ACI (14.7)")
    controles(T, comun)

    # ---------------------------------------------------------- 2. Intervalos propios ---
    print("\n2. Intervalos propios del 90 % y rejilla de 19 cuantiles")
    filas, horas_is, horas_crps = [], {}, {}
    for m in MODELOS:
        lo, hi = propios[m]
        Q = T[m][P.COLUMNAS_Q].to_numpy(float)
        horas_is[m] = P.puntuacion_intervalo(y, lo, hi)
        horas_crps[m] = P.crps19(y, Q)
        for nombre, mask in (("E", E), ("8760", TODAS)):
            filas.append({"modelo": m, "conjunto": nombre, **metricas_modelo(y, lo, hi, Q, mask, comun),
                          "MAE_prevision": np.abs(y - T[m]["prevision"].to_numpy())[mask].mean()})
    propios_t = pd.DataFrame(filas)
    propios_t.to_csv(DIR / "propios_metricas.csv", index=False)
    for conjunto in ("8760", "E"):
        t = propios_t[propios_t["conjunto"] == conjunto].set_index("modelo")
        print(f"\n  Conjunto {conjunto}:")
        cols = ["cobertura", "cob_ic_inf", "cob_ic_sup", "fallo_debajo", "fallo_encima",
                "cociente_fallos", "ancho_medio", "ancho_mediana", "ancho_relativo", "IS", "IS_ic_inf",
                "IS_ic_sup", "pinball05", "pinball95", "CRPS19", "CRPS19_ic_inf", "CRPS19_ic_sup",
                "fiab_desv_max", "MAE_prevision"]
        print(t[cols].T.to_string(float_format=lambda x: f"{x:.4g}"))
        print("  Cobertura por bloque (B1 12-17 h, B2 18-23, B3 24-29, B4 30-36):")
        print(t[[f"cob_B{b}" for b in range(1, 5)]].round(3).to_string())
        print("  Cobertura por mes:")
        print(t[[f"cob_mes{k:02d}" for k in range(1, 13)]].round(2).to_string())

    # Fiabilidad de la rejilla propia
    fiab = pd.DataFrame({(m, c): P.fiabilidad(y[mask], T[m][P.COLUMNAS_Q].to_numpy(float)[mask])
                         for m in MODELOS for c, mask in (("E", E), ("8760", TODAS))}, index=P.NIVELES)
    fiab.index.name = "tau"
    fiab.to_csv(DIR / "fiabilidad_propios.csv")
    print("\n  Fiabilidad de la rejilla propia en 8.760 horas (frecuencia observada de real ≤ q_τ):")
    print(fiab.xs("8760", axis=1, level=1).round(3).T.to_string())

    # Diferencias pareadas entre modelos
    pares = []
    for i, a in enumerate(MODELOS):
        for b in MODELOS[i + 1:]:
            for nombre, h in (("IS", horas_is), ("CRPS19", horas_crps)):
                for conjunto, mask in (("E", E), ("8760", TODAS)):
                    media, lo_, hi_, lectura = diferencia(h[a][mask], h[b][mask], dia[mask])
                    pares.append({"a": a, "b": b, "metrica": nombre, "conjunto": conjunto,
                                  "diferencia_a_menos_b": media, "ic_inf": lo_, "ic_sup": hi_, "lectura": lectura})
    pares = pd.DataFrame(pares)
    pares.to_csv(DIR / "propios_pares.csv", index=False)
    print("\n  Diferencia pareada a − b (negativa = a mejor) en E, IC del 95 % por bootstrap de días:")
    print(pares[pares["conjunto"] == "E"].drop(columns="conjunto").round(1).to_string(index=False))

    # -------------------------------------------------------------------------- 3. ACI ---
    print("\n3. ACI con MAPIE (W = 336, un conjunto por bloque, retroalimentación retrasada)")
    configs = [(P.ALFA, 0.0)] + [(P.ALFA, g) for g in P.GAMMAS]
    gammas_todas = [0.0, *P.GAMMAS]
    resultados, tabla_is, filas_g = {}, {}, []
    for m in MODELOS:
        e = entradas_de(m, T, comun)
        t0 = time.perf_counter()
        r = P.motor_aci(e["hora_utc"], e["origen_utc"], e["real"], e["prevision"], e["bloque"],
                        e["dia"], configs, P.DIAS_ARRANQUE_2023)
        resultados[m] = r
        tabla_is[m] = {}
        for c, g in enumerate(gammas_todas):
            lo, hi = r["inferior"][c], r["superior"][c]
            mm = metricas_modelo(y, lo, hi, None, E, comun)
            tabla_is[m][g] = mm["IS"]
            filas_g.append({"modelo": m, "gamma": g, "cobertura": mm["cobertura"], "ancho_medio": mm["ancho_medio"],
                            "IS": mm["IS"], "horas_infinitas": int(r["horas_infinitas"][c]),
                            "alfa_t_medio": float(np.nanmean(r["alfa_emitido"][c][E]))})
        print(f"  {NOMBRE[m]}: ACI en {len(configs)} configuraciones ({time.perf_counter() - t0:.0f} s)", flush=True)
    tg = pd.DataFrame(filas_g)
    tg["IS_rel_g0"] = tg["IS"] / tg.groupby("modelo")["IS"].transform(lambda s: s.iloc[0])
    tg.to_csv(DIR / "aci_gamma.csv", index=False)
    print("\n  Por γ (en E): cobertura, anchura media, IS e IS relativo a γ = 0:")
    print(tg.round({"cobertura": 3, "ancho_medio": 0, "IS": 0, "IS_rel_g0": 4, "alfa_t_medio": 3}).to_string(index=False))
    gamma, puntuacion = elegir_gamma(tabla_is)
    print("\n  Puntuación de la regla 14.6.7 (media de IS_m(γ)/IS_m(0)): "
          + ", ".join(f"γ {g}: {p:.4f}" for g, p in puntuacion.items()))
    print(f"  γ ELEGIDO: {gamma}" + ("" if min(puntuacion.values()) < 1 else
          "  (RESULTADO NEGATIVO: ningún candidato queda por debajo de 1)"))

    # Cuantiles de ACI con el γ elegido y métricas finales
    print("\n  Rejilla de cuantiles con ACI (9 niveles centrales) y métricas con el γ elegido:")
    c_elegido = gammas_todas.index(gamma)
    filas_aci, QACI, horas_is_aci, horas_crps_aci = [], {}, {}, {}
    for m in MODELOS:
        e = entradas_de(m, T, comun)
        Q, reordenadas, infinitas = P.cuantiles_de_aci(e["hora_utc"], e["origen_utc"], e["real"], e["prevision"],
                                                       e["bloque"], e["dia"], P.DIAS_ARRANQUE_2023, gamma)
        QACI[m] = Q
        r = resultados[m]
        lo, hi = r["inferior"][c_elegido], r["superior"][c_elegido]
        coincide = float(np.mean((Q[E, 0] == lo[E]) & (Q[E, 18] == hi[E])))
        horas_is_aci[m] = P.puntuacion_intervalo(y, lo, hi)
        horas_crps_aci[m] = P.crps19(y, np.where(np.isnan(Q), 0.0, Q))
        mm = metricas_modelo(y, lo, hi, Q, E, comun)
        inf90 = int(r["horas_infinitas"][c_elegido])
        filas_aci.append({"modelo": m, "gamma": gamma, **mm, "horas_infinitas_90": inf90,
                          "pct_infinitas_90": 100 * inf90 / int(E.sum()), "horas_reordenadas": reordenadas,
                          "horas_infinitas_niveles": int(infinitas.sum()), "nivel_09_coincide_con_90": coincide})
        out = pd.DataFrame({"hora_utc": T[m]["hora_utc"], "fecha_objetivo": T[m]["fecha_objetivo"],
                            "real": T[m]["real"], "prevision": T[m]["prevision"]})
        for c, g in enumerate(gammas_todas):
            out[f"inf_g{g}"], out[f"sup_g{g}"] = r["inferior"][c], r["superior"][c]
        out["alfa_t"] = r["alfa_emitido"][c_elegido]
        for j, col in enumerate(P.COLUMNAS_Q):
            out[f"aci_{col}"] = Q[:, j]
        out.to_parquet(v.DIR_PREDICCIONES / f"{m}_aci_validacion.parquet", index=False)
    aci_t = pd.DataFrame(filas_aci)
    aci_t.to_csv(DIR / "aci_metricas.csv", index=False)
    cols = ["cobertura", "cob_ic_inf", "cob_ic_sup", "fallo_debajo", "fallo_encima", "cociente_fallos",
            "ancho_medio", "ancho_mediana", "IS", "IS_ic_inf", "IS_ic_sup", "CRPS19", "CRPS19_ic_inf",
            "CRPS19_ic_sup", "fiab_desv_max", "horas_infinitas_90", "pct_infinitas_90", "horas_reordenadas",
            "nivel_09_coincide_con_90"]
    print(aci_t.set_index("modelo")[cols].T.to_string(float_format=lambda x: f"{x:.4g}"))
    print("  Cobertura de ACI por bloque y por mes:")
    print(aci_t.set_index("modelo")[[f"cob_B{b}" for b in range(1, 5)]].round(3).to_string())
    print(aci_t.set_index("modelo")[[f"cob_mes{k:02d}" for k in range(1, 13)]].round(2).to_string())

    # Propio frente a ACI en E (mismas horas), diferencia pareada
    comp = []
    for m in MODELOS:
        for nombre, hp, ha in (("IS", horas_is[m], horas_is_aci[m]), ("CRPS19", horas_crps[m], horas_crps_aci[m])):
            media, lo_, hi_, lectura = diferencia(ha[E], hp[E], dia[E])
            comp.append({"modelo": m, "metrica": nombre, "aci_menos_propio": media, "ic_inf": lo_, "ic_sup": hi_,
                         "lectura": lectura, "propio": hp[E].mean(), "aci": ha[E].mean()})
        pf = propios_t[(propios_t["modelo"] == m) & (propios_t["conjunto"] == "E")].iloc[0]
        af = aci_t[aci_t["modelo"] == m].iloc[0]
        comp.append({"modelo": m, "metrica": "|cobertura-0,90|", "propio": abs(pf["cobertura"] - 0.9),
                     "aci": abs(af["cobertura"] - 0.9),
                     "lectura": ("mejora" if (abs(pf["cobertura"] - 0.9) - abs(af["cobertura"] - 0.9) >= 0.01
                                              or (not 0.88 <= pf["cobertura"] <= 0.92
                                                  and 0.88 <= af["cobertura"] <= 0.92)) else "sin mejora")})
    comp = pd.DataFrame(comp)
    comp.to_csv(DIR / "aci_vs_propio.csv", index=False)
    print("\n  ACI frente al intervalo propio en E (diferencia pareada ACI − propio; negativa = ACI mejor):")
    print(comp.round({"aci_menos_propio": 1, "ic_inf": 1, "ic_sup": 1, "propio": 3, "aci": 3}).to_string(index=False))
    fiab_aci = pd.DataFrame({m: P.fiabilidad(y[E], QACI[m][E]) for m in MODELOS}, index=P.NIVELES)
    fiab_aci.index.name = "tau"
    fiab_aci.to_csv(DIR / "fiabilidad_aci.csv")
    print("\n  Fiabilidad de la rejilla con ACI en E:")
    print(fiab_aci.round(3).T.to_string())

    figuras(T, propios, comun, QACI, resultados, c_elegido, E, fiab, fiab_aci)
    anticipacion(propios_t, aci_t, comp, pares, gamma, fiab)
    print("\nListo. Tablas y figuras en", DIR.relative_to(v.RAIZ))


def figuras(T, propios, comun, QACI, resultados, c_elegido, E, fiab, fiab_aci):
    """Fiabilidad, cobertura por bloque y mes, y una semana fija (14.5)."""
    y = comun["real"]
    fig, ejes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)
    for ax, m in zip(ejes, MODELOS):
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="calibración perfecta")
        ax.plot(P.NIVELES, fiab[(m, "E")], marker="o", ms=3, label="propia (E)")
        ax.plot(P.NIVELES, fiab_aci[m], marker="s", ms=3, label="con ACI (E)")
        ax.set_title(NOMBRE[m]); ax.set_xlabel("nivel τ del cuantil"); ax.grid(alpha=0.3)
    ejes[0].set_ylabel("frecuencia observada de real ≤ q_τ"); ejes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(DIR / "fiabilidad.png", dpi=130); plt.close(fig)

    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.2))
    for m in MODELOS:
        lo, hi = propios[m]
        dentro = (lo <= y) & (y <= hi)
        ejes[0].plot(range(1, 5), [dentro[E & (comun["bloque"] == b)].mean() for b in range(4)], marker="o", label=NOMBRE[m])
        ejes[1].plot(range(1, 13), [dentro[E & (comun["mes"] == k)].mean() if (E & (comun["mes"] == k)).any() else np.nan
                                    for k in range(1, 13)], marker="o", label=NOMBRE[m])
    for ax, titulo, etiqueta in zip(ejes, ("por bloque de anticipación", "por mes"), ("bloque (B1 12-17 h … B4 30-36 h)", "mes de 2023")):
        ax.axhline(0.9, color="0.4", ls="--", lw=1); ax.set_title(f"Cobertura propia del 90 % en E, {titulo}")
        ax.set_xlabel(etiqueta); ax.set_ylabel("cobertura"); ax.grid(alpha=0.3)
    ejes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(DIR / "cobertura_bloque_mes.png", dpi=130); plt.close(fig)

    fechas = pd.to_datetime(T["ingenuo"]["fecha_objetivo"])
    sel = ((fechas >= SEMANA[0]) & (fechas <= SEMANA[1])).to_numpy()
    x = np.arange(sel.sum())
    fig, ejes = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey=True)
    for ax, m in zip(ejes.ravel(), MODELOS):
        lo, hi = propios[m]
        ax.fill_between(x, lo[sel], hi[sel], alpha=0.25, label="propio 90 %")
        ax.fill_between(x, resultados[m]["inferior"][c_elegido][sel], resultados[m]["superior"][c_elegido][sel],
                        alpha=0.25, color="tab:orange", label="ACI 90 %")
        ax.plot(x, T[m]["prevision"].to_numpy()[sel], lw=1, color="k", label="previsión")
        ax.plot(x, y[sel], lw=1, color="tab:red", label="real"); ax.set_title(NOMBRE[m]); ax.grid(alpha=0.3)
    ejes[0, 0].legend(fontsize=8); fig.supxlabel(f"horas del {SEMANA[0]} al {SEMANA[1]} (días objetivo)")
    fig.supylabel("MW"); fig.tight_layout(); fig.savefig(DIR / "semana_ejemplo.png", dpi=130); plt.close(fig)


def anticipacion(propios_t, aci_t, comp, pares, gamma, fiab):
    """Tabla automática de 14.9: cifra anticipada frente a la observada (✓ dentro del rango)."""
    t = propios_t[propios_t["conjunto"] == "8760"].set_index("modelo")
    filas = []

    def rango(nombre, valor, bajo, alto, unidad=""):
        filas.append((nombre, f"{bajo:g} a {alto:g}{unidad}", f"{valor:.3g}{unidad}", "✓" if bajo <= valor <= alto else "✗ FALLA"))

    for m, (b, a) in {"ingenuo": (86, 92), "sarimax": (82, 91), "lightgbm": (68, 86), "nhits": (78, 92)}.items():
        rango(f"cobertura propia 8760, {NOMBRE[m]}", 100 * t.loc[m, "cobertura"], b, a, " %")
    filas.append(("LightGBM tiene la cobertura más baja y < 85 %", "sí",
                  "sí" if (t["cobertura"].idxmin() == "lightgbm" and t.loc["lightgbm", "cobertura"] < 0.85) else "no",
                  "✓" if (t["cobertura"].idxmin() == "lightgbm" and t.loc["lightgbm", "cobertura"] < 0.85) else "✗ FALLA"))
    for m in ("lightgbm", "sarimax"):
        filas.append((f"cociente fallos debajo/encima > 2, {NOMBRE[m]}", "> 2", f"{t.loc[m, 'cociente_fallos']:.2f}",
                      "✓" if t.loc[m, "cociente_fallos"] > 2 else "✗ FALLA"))
    filas.append(("cociente fallos N-HiTS < 0,5 («al revés»)", "< 0,5", f"{t.loc['nhits', 'cociente_fallos']:.2f}",
                  "✓" if t.loc["nhits", "cociente_fallos"] < 0.5 else "✗ FALLA"))
    rango("cociente fallos ingenuo", t.loc["ingenuo", "cociente_fallos"], 0.6, 1.6)
    for m, (b, a) in {"ingenuo": (5200, 7000), "sarimax": (3600, 5800), "lightgbm": (1500, 2900), "nhits": (2200, 4500)}.items():
        rango(f"anchura media {NOMBRE[m]}", t.loc[m, "ancho_medio"], b, a, " MW")
    for m, (b, a) in {"ingenuo": (8500, 10000), "sarimax": (6300, 7500), "lightgbm": (4000, 5000), "nhits": (4700, 5500)}.items():
        rango(f"IS propio {NOMBRE[m]}", t.loc[m, "IS"], b, a, " MW")
    orden_is = list(t["IS"].sort_values().index)
    orden_crps = list(t["CRPS19"].sort_values().index)
    esperado = ["lightgbm", "nhits", "sarimax", "ingenuo"]
    filas.append(("orden por IS (mejor→peor)", " < ".join(esperado), " < ".join(orden_is), "✓" if orden_is == esperado else "✗ FALLA"))
    filas.append(("orden por CRPS₁₉ (mejor→peor)", " < ".join(esperado), " < ".join(orden_crps), "✓" if orden_crps == esperado else "✗ FALLA"))
    for m in MODELOS:
        rango(f"CRPS₁₉/MAE {NOMBRE[m]}", t.loc[m, "CRPS19"] / t.loc[m, "MAE_prevision"], 0.70, 0.85)
    for m, (b, a) in {"ingenuo": (0, 0.04), "sarimax": (0.03, 0.08), "lightgbm": (0.06, 0.14), "nhits": (0.03, 0.10)}.items():
        rango(f"desviación máx. de fiabilidad {NOMBRE[m]}", t.loc[m, "fiab_desv_max"], b, a)
    f8 = fiab.xs("8760", axis=1, level=1)
    for m in ("lightgbm", "sarimax"):
        filas.append((f"fiabilidad por encima de la diagonal, {NOMBRE[m]}", "media(frec − τ) > 0",
                      f"{(f8[m] - f8.index.to_numpy()).mean():+.3f}", "✓" if (f8[m] - f8.index.to_numpy()).mean() > 0 else "✗ FALLA"))
    b3 = propios_t[(propios_t["conjunto"] == "8760") & (propios_t["modelo"] == "lightgbm")].iloc[0]
    cobs = [b3[f"cob_B{k}"] for k in range(1, 5)]
    filas.append(("cobertura propia de LightGBM más baja en B3", "B3", f"B{int(np.argmin(cobs)) + 1}",
                  "✓" if int(np.argmin(cobs)) == 2 else "✗ FALLA"))
    filas.append(("γ elegido", "0,005 o 0,01", f"{gamma}", "✓" if gamma in (0.005, 0.01) else "✗ FALLA"))
    a = aci_t.set_index("modelo")
    for m in MODELOS:
        rango(f"cobertura ACI en E, {NOMBRE[m]}", 100 * a.loc[m, "cobertura"], 88, 92, " %")
    rango("ancho medio de ACI, LightGBM", a.loc["lightgbm", "ancho_medio"], 2700, 3500, " MW")
    for m in MODELOS:
        r = comp[(comp["modelo"] == m) & (comp["metrica"] == "IS")].iloc[0]
        if m == "lightgbm":
            rel = 100 * r["aci_menos_propio"] / r["propio"]
            filas.append(("IS de ACI frente al propio, LightGBM", "−20 a −5 % y apreciable", f"{rel:+.1f} % ({r['lectura']})",
                          "✓" if (-20 <= rel <= -5 and r["lectura"] == "apreciable") else "✗ FALLA"))
        else:
            filas.append((f"IS de ACI frente al propio, {NOMBRE[m]}", "no distinguible", r["lectura"],
                          "✓" if r["lectura"] == "no distinguible" else "✗ FALLA"))
    filas.append(("horas con límite infinito < 1 % (todos)", "< 1 %", f"máx. {a['pct_infinitas_90'].max():.2f} %",
                  "✓" if a["pct_infinitas_90"].max() < 1 else "✗ FALLA"))
    res = pd.DataFrame(filas, columns=["anticipación (14.9)", "anticipado", "observado", "resultado"])
    res.to_csv(DIR / "anticipacion_vs_resultado.csv", index=False)
    print("\n5. Anticipación de 14.9 frente a lo observado")
    print(res.to_string(index=False))
    print(f"  Fallan {int(res['resultado'].str.contains('FALLA').sum())} de {len(res)} comprobaciones automáticas.")


if __name__ == "__main__":
    main()
