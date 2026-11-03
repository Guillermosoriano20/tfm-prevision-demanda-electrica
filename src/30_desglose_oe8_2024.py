"""Desglose de OE8 en 2024 (protocolo 16). SE EJECUTA UNA SOLA VEZ.

Lee las previsiones, los intervalos y los resultados de ACI de 2024 que dejaron 28 y 29 y los desglosa
por hora local, bloque de 6 h, tipo de día, cruce tipo de día x bloque, día de la semana y periodo
navideño: (1) controles (se detiene si falla alguno), incluida la reproducción de las cifras por bloque
que ya imprimió el 29; (2) precisión (MAE, sesgo, RMSE); (3) modelo preferible por celda y contraste del
cambio de régimen (16.7); (4) calibración de los intervalos propios, de la ventana (gamma = 0) y de ACI
(gamma = 0,001); (5) peores días, puente, domingo festivo y Navidad; (6) figuras; (7) la anticipación
A38-A52 de 16.8 frente a lo observado.

No entrena, no reajusta y no decide nada (16.1.2). Candado: si ya existe
data/resultados/final/desglose/manifiesto.json, el script se niega a ejecutarse salvo con
--repetir-por-error "motivo".
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
from exogenas import navidad  # noqa: E402

d24 = F.cargar_script("24_validacion_desgloses.py")      # estilo de figuras y formato numérico
MODELOS, NOMBRE = F.MODELOS, F.NOMBRE
DIR = F.DIR_RESULTADOS / "desglose"
DIR_29 = F.DIR_RESULTADOS
REF = "lightgbm"                                          # el preferible global (15.14.7)
TIPOS = ("laborable", "fin_de_semana", "festivo")         # el puente va aparte (16.3)
BLOQUES = ["B1 00-05 h", "B2 06-11 h", "B3 12-17 h", "B4 18-23 h"]
MIN_DIAS = 10                                             # mínimo de días para afirmar un cambio (16.7.2)
PUENTE_FECHA = "2024-08-16"
DOMINGO_FESTIVO = "2024-12-08"
TIPOS_INT = ("propio", "ventana", "aci")


# ------------------------------------------------------------------------------ candado ---
def candado(repetir):
    ruta = DIR / "manifiesto.json"
    DIR.mkdir(parents=True, exist_ok=True)
    manifiesto = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None
    if manifiesto and not repetir:
        raise SystemExit("El desglose de OE8 ya se ejecutó (manifiesto.json). Solo se repite por un error de "
                         "programación: --repetir-por-error \"motivo\" (protocolo 16.1.3).")
    nuevo = F.manifiesto({"estado": "iniciada"})
    if manifiesto:
        nuevo["repeticiones"] = manifiesto.get("repeticiones", []) + [
            {"motivo": repetir, "ejecucion_anterior": {k: manifiesto[k] for k in ("fecha", "git") if k in manifiesto}}]
    ruta.write_text(json.dumps(nuevo, indent=1, default=str), encoding="utf-8")
    return nuevo


def cerrar_manifiesto(nuevo, extra):
    nuevo.update(estado="completa", **extra)
    (DIR / "manifiesto.json").write_text(json.dumps(nuevo, indent=1, default=str), encoding="utf-8")


# --------------------------------------------------------------------------------- datos ---
def cargar():
    """Previsiones, rejillas, intervalos y calendario de las 8.784 horas de 2024, con los controles de 16.4.1."""
    datos = v.cargar_datos(incluir_prueba=True)
    T = {m: pd.read_parquet(v.DIR_PREDICCIONES / f"{m}_cuantiles_prueba.parquet") for m in MODELOS}
    A = {m: pd.read_parquet(v.DIR_PREDICCIONES / f"{m}_aci_prueba.parquet") for m in MODELOS}
    ref = T["ingenuo"]
    for m in MODELOS:
        for t in (T[m], A[m]):
            assert len(t) == 8784 and not t.isna().any().any(), f"{m}: filas o nulos"
            assert np.array_equal(P.a_ns(t["hora_utc"]), P.a_ns(ref["hora_utc"])), f"{m}: horas distintas"
            assert np.array_equal(t["real"].to_numpy(), ref["real"].to_numpy()), f"{m}: `real` distinta"
        assert np.array_equal(A[m]["prevision"].to_numpy(), T[m]["prevision"].to_numpy()), f"{m}: previsión distinta en ACI"
    idx = pd.DatetimeIndex(ref["hora_utc"]).as_unit(datos.index.unit)
    cal = datos.reindex(idx)
    assert np.array_equal(cal["demanda_mw"].to_numpy(), ref["real"].to_numpy()), "`real` distinta del dataset"
    cal = cal.reset_index(drop=True)
    cal["navidad"] = navidad(cal).astype(bool)
    _, bloque = P.anticipacion_y_bloque(ref, v.generar_origenes("prueba"))      # el mismo bloque que usó el 29
    fechas = pd.to_datetime(ref["fecha_objetivo"]).reset_index(drop=True)
    comun = {"y": ref["real"].to_numpy(float), "bloque": bloque, "dia": pd.factorize(fechas)[0],
             "hora": cal["hora_del_dia"].to_numpy(int), "tipo": cal["tipo_dia"].to_numpy(str),
             "dsem": cal["dia_semana"].to_numpy(int), "navidad": cal["navidad"].to_numpy(bool),
             "fecha": fechas.dt.strftime("%Y-%m-%d").to_numpy(), "frac": cal["fraccion_festivo"].to_numpy(float)}
    prev, absE, sesgoE, intervalos, rejillas = {}, {}, {}, {}, {}
    for m in MODELOS:
        prev[m] = T[m]["prevision"].to_numpy(float)
        e = comun["y"] - prev[m]
        absE[m], sesgoE[m] = np.abs(e), e
        if m in ("lightgbm", "nhits"):
            o = pd.read_parquet(v.DIR_PREDICCIONES / f"{m}_intervalos_prueba.parquet")
            assert np.array_equal(o["prevision"].to_numpy(), T[m]["prevision"].to_numpy())
            propio = (o["inferior"].to_numpy(float), o["superior"].to_numpy(float))
        else:
            propio = (T[m]["q05"].to_numpy(float), T[m]["q95"].to_numpy(float))
        intervalos[m] = {"propio": propio,
                         "ventana": (A[m]["inf_g0.0"].to_numpy(float), A[m]["sup_g0.0"].to_numpy(float)),
                         "aci": (A[m]["inf_g0.001"].to_numpy(float), A[m]["sup_g0.001"].to_numpy(float))}
        rejillas[m] = {"propio": T[m][P.COLUMNAS_Q].to_numpy(float),
                       "aci": A[m][["aci_" + c for c in P.COLUMNAS_Q]].to_numpy(float)}
    return comun, prev, absE, sesgoE, intervalos, rejillas


def celdas(c):
    """familia -> {etiqueta: máscara}. El puente va dentro de «tipo» pero marcado como descriptivo."""
    h, b, t = c["hora"], c["bloque"], c["tipo"]
    fam = {"hora": {f"{k:02d} h": h == k for k in range(24)},
           "bloque": {BLOQUES[k]: b == k for k in range(4)},
           "tipo": {k: t == k for k in (*TIPOS, "puente")},
           "cruce": {f"{k} | {BLOQUES[j]}": (t == k) & (b == j) for k in TIPOS for j in range(4)},
           "dia_semana": {d24.DIAS[k]: c["dsem"] == k for k in range(7)},
           "navidad": {"dentro (23/12-6/1)": c["navidad"], "fuera": ~c["navidad"]}}
    return fam


# ------------------------------------------------------------------------------ controles ---
def controles(c, fam, absE, intervalos):
    t0 = time.perf_counter()
    dias_por_tipo = {k: len(np.unique(c["dia"][m])) for k, m in fam["tipo"].items()}
    print(f"  16.4.2 días por tipo: {dias_por_tipo}; horas por hora local: "
          f"{sorted(set(int(m.sum()) for m in fam['hora'].values()))}")
    if not F.ARNES:
        assert dias_por_tipo == {"laborable": 253, "fin_de_semana": 102, "festivo": 10, "puente": 1}, "recuentos de 16.3"
    assert all(int(m.sum()) == 366 for m in fam["hora"].values()), "no hay 366 observaciones por hora local"
    assert c["fecha"][c["tipo"] == "puente"][0] == PUENTE_FECHA or F.ARNES
    # 16.4.4: las particiones suman 8.784 horas y reproducen el MAE global
    for nombre in ("hora", "bloque", "tipo"):
        assert sum(int(m.sum()) for m in fam[nombre].values()) == 8784, f"las celdas de {nombre} no suman 8.784 horas"
    assert sum(int(m.sum()) for m in fam["cruce"].values()) == 8784 - int(fam["tipo"]["puente"].sum())
    for m in MODELOS:
        for nombre in ("hora", "bloque", "tipo"):
            mae = sum(absE[m][k].sum() for k in fam[nombre].values()) / 8784
            assert abs(mae - absE[m].mean()) < 0.05, f"{m}: MAE global por {nombre}"
    # 16.4.3: reproducción de lo que ya imprimió el 29 (se compara con sus CSV, a precisión completa)
    mb = pd.read_csv(DIR_29 / "puntuales_mes_bloque.csv")
    for m in MODELOS:
        ref_b = mb[(mb.modelo == m) & (mb.eje == "bloque")].sort_values("valor")["MAE"].to_numpy()
        mio = np.array([absE[m][fam["bloque"][b]].mean() for b in BLOQUES])
        assert np.allclose(mio, ref_b, atol=1e-6), f"{m}: MAE por bloque distinto del 29"
    tablas = {"propio": pd.read_csv(DIR_29 / "propios_metricas.csv", index_col=0)}
    for nombre, archivo, g in (("ventana", "ventana_metricas.csv", 0.0), ("aci", "aci_metricas.csv", 0.001)):
        t = pd.read_csv(DIR_29 / archivo)
        tablas[nombre] = t[np.isclose(t["gamma"], g)].set_index("modelo")
    for tipo, t in tablas.items():
        for m in MODELOS:
            lo, hi = intervalos[m][tipo]
            dentro = (lo <= c["y"]) & (c["y"] <= hi)
            for j, b in enumerate(BLOQUES):
                assert abs(dentro[fam["bloque"][b]].mean() - t.loc[m, f"cob_B{j + 1}"]) < 1e-9, \
                    f"{m} {tipo}: cobertura por bloque distinta del 29"
            assert abs(dentro.mean() - t.loc[m, "cobertura"]) < 1e-9, f"{m} {tipo}: cobertura global distinta del 29"
    print("  16.4.3 MAE por bloque y cobertura (propia, ventana, ACI) por bloque y global: iguales a las del 29")
    # 16.4.5: el contraste por bootstrap, con una diferencia sintética conocida
    rng = np.random.default_rng(1)
    dias = np.repeat(np.arange(100), 24)
    centrada = rng.normal(0, 50, 100)[dias] + rng.normal(0, 100, dias.size)
    desplazada = centrada + 60
    pc = [contraste_p(rng.normal(0, 50, 100)[dias] + rng.normal(0, 100, dias.size), dias)[1] for _ in range(60)]
    frac = float(np.mean(np.array(pc) >= 0.05))
    p_des = contraste_p(desplazada, dias)[1]
    print(f"  16.4.5 contraste sintético: sin diferencia, p ≥ 0,05 en {100 * frac:.0f} % de 60 repeticiones; con +60, p = {p_des:.4f}")
    assert 0.85 <= frac <= 1.0 and p_des < 0.05, "falla el control del contraste"
    print(f"  controles superados ({time.perf_counter() - t0:.0f} s)", flush=True)


# ------------------------------------------------------------------------------- contraste ---
def contraste_p(diferencia, dias):
    """(IC 95 % percentil, p bilateral por bootstrap de días) de la media de una diferencia pareada por horas."""
    r = P.media_bootstrap(diferencia, dias)
    n = len(r)
    p = 2 * min((np.sum(r <= 0) + 1) / (n + 1), (np.sum(r >= 0) + 1) / (n + 1))
    return (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))), min(1.0, float(p))


# ------------------------------------------------------------------------------- precisión ---
def tabla_precision(c, fam, absE, sesgoE):
    filas = []
    for nombre, grupo in fam.items():
        for etiqueta, mask in grupo.items():
            n_dias = len(np.unique(c["dia"][mask]))
            mae_ing = absE["ingenuo"][mask].mean()
            for m in MODELOS:
                filas.append({"familia": nombre, "celda": etiqueta, "modelo": m, "n_horas": int(mask.sum()),
                              "n_dias": n_dias, "MAE": absE[m][mask].mean(), "sesgo": sesgoE[m][mask].mean(),
                              "RMSE": float(np.sqrt((sesgoE[m][mask] ** 2).mean())),
                              "MAE_rel_ingenuo": absE[m][mask].mean() / mae_ing})
    return pd.DataFrame(filas)


def preferibles(c, fam, absE, precision):
    """Modelo de menor MAE por celda y, en bloques, tipos de día y cruces, el contraste de 16.7."""
    filas = []
    for nombre, grupo in fam.items():
        for etiqueta, mask in grupo.items():
            mae = {m: absE[m][mask].mean() for m in MODELOS}
            orden = sorted(mae, key=mae.get)
            mejor, segundo = orden[0], orden[1]
            comparado = segundo if mejor == REF else mejor
            fila = {"familia": nombre, "celda": etiqueta, "n_dias": len(np.unique(c["dia"][mask])), "mejor": mejor,
                    "MAE_mejor": mae[mejor], "segundo": segundo, "MAE_segundo": mae[segundo],
                    "comparado_con_LightGBM": comparado, "dif_LightGBM_menos_comparado": mae[REF] - mae[comparado]}
            fila["con_contraste"] = nombre in ("bloque", "tipo", "cruce") and fila["n_dias"] >= MIN_DIAS
            if fila["con_contraste"]:
                (lo, hi), p = contraste_p((absE[REF] - absE[comparado])[mask], c["dia"][mask])
                fila.update(ic_inf=lo, ic_sup=hi, p=p)
            filas.append(fila)
    t = pd.DataFrame(filas)
    t["p_holm"] = np.nan
    for nombre in ("bloque", "tipo", "cruce"):
        sel = (t.familia == nombre) & t.con_contraste
        t.loc[sel, "p_holm"] = F.holm(t.loc[sel, "p"].to_numpy())
    def lectura(r):
        if not r.con_contraste:
            return "descriptivo"
        apreciable = r.p_holm < 0.05
        if r.mejor == REF:
            return "LightGBM se mantiene (apreciable)" if apreciable else "LightGBM se mantiene (no distinguible del segundo)"
        return "CAMBIO DE RÉGIMEN" if apreciable else "mejor nominal distinto, no distinguible"
    t["lectura"] = t.apply(lectura, axis=1)
    return t


# ------------------------------------------------------------------------------ calibración ---
def tabla_calibracion(c, fam, intervalos, rejillas):
    filas = []
    y = c["y"]
    for nombre in ("hora", "bloque", "tipo", "cruce"):
        for etiqueta, mask in fam[nombre].items():
            dias = c["dia"][mask]
            n_dias = len(np.unique(dias))
            con_ic = nombre != "hora" and etiqueta != "puente"
            for m in MODELOS:
                for tipo in TIPOS_INT:
                    lo, hi = intervalos[m][tipo]
                    dentro = ((lo <= y) & (y <= hi))[mask]
                    por_debajo, por_encima = (y[mask] < lo[mask]).mean(), (y[mask] > hi[mask]).mean()
                    f = {"familia": nombre, "celda": etiqueta, "modelo": m, "intervalo": tipo, "n_horas": int(mask.sum()),
                         "n_dias": n_dias, "cobertura": dentro.mean(),
                         "fallo_debajo": por_debajo, "fallo_encima": por_encima,
                         "cociente_fallos": por_debajo / por_encima if por_encima > 0 else np.inf,
                         "ancho_medio": (hi - lo)[mask].mean(),
                         "IS": P.puntuacion_intervalo(y[mask], lo[mask], hi[mask]).mean()}
                    if con_ic:
                        f["cob_ic_inf"], f["cob_ic_sup"] = P.ic95(dentro.astype(float), dias)[1:]
                    if tipo in rejillas[m]:
                        f["CRPS19"] = P.crps19(y[mask], rejillas[m][tipo][mask]).mean()
                    filas.append(f)
    return pd.DataFrame(filas)


# ------------------------------------------------------------------------------- por días ---
def dias_y_periodos(c, absE, sesgoE):
    d = pd.DataFrame({"fecha": c["fecha"], "tipo": c["tipo"], "dsem": c["dsem"], "navidad": c["navidad"],
                      "frac": c["frac"]})
    for m in MODELOS:
        d[f"MAE_{m}"] = absE[m]
        d[f"sesgo_{m}"] = sesgoE[m]
    dia = d.groupby("fecha").agg({"tipo": "first", "dsem": "first", "navidad": "first", "frac": "first",
                                  **{f"MAE_{m}": "mean" for m in MODELOS}, **{f"sesgo_{m}": "mean" for m in MODELOS}})
    dia["dia_semana"] = dia["dsem"].map(dict(enumerate(d24.DIAS)))
    peores = {}
    for m in MODELOS:
        p = dia.sort_values(f"MAE_{m}", ascending=False).head(10)[["dia_semana", "tipo", "frac", "navidad", f"MAE_{m}"]]
        peores[m] = p.rename(columns={f"MAE_{m}": "MAE_dia"})
    return dia, peores


# --------------------------------------------------------------------------------- figuras ---
def figuras(c, fam, absE, calib):
    d24.plt.rcParams.update({"font.size": 9})
    x = np.arange(24)
    fig, ax = plt.subplots(figsize=(d24.ANCHO, 3.6))
    for m in MODELOS:
        color, marca = d24.ESTILO[m]
        ax.plot(x, [absE[m][fam["hora"][f"{k:02d} h"]].mean() for k in x], color=color, marker=marca, ms=3, label=NOMBRE[m])
    ax.set_xlabel("hora local del día objetivo"); ax.set_ylabel("MAE (MW)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(d24.FuncFormatter(d24.numero_es))
    fig.tight_layout(); fig.savefig(DIR / "mae_por_hora.png", dpi=300); plt.close(fig)
    fig, ejes = plt.subplots(1, 2, figsize=(d24.ANCHO * 1.7, 3.6), sharey=True)
    for ax, tipo, titulo in zip(ejes, ("propio", "aci"), ("intervalo propio", "con ACI (γ = 0,001)")):
        for m in MODELOS:
            color, marca = d24.ESTILO[m]
            s = calib[(calib.familia == "hora") & (calib.modelo == m) & (calib.intervalo == tipo)].sort_values("celda")
            ax.plot(x, s["cobertura"].to_numpy(), color=color, marker=marca, ms=3, label=NOMBRE[m])
        ax.axhline(0.9, color="0.4", ls="--", lw=1); ax.set_title(titulo); ax.grid(alpha=0.3)
        ax.set_xlabel("hora local del día objetivo")
    ejes[0].set_ylabel("cobertura del 90 %"); ejes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(DIR / "cobertura_por_hora.png", dpi=300); plt.close(fig)


# ---------------------------------------------------------------------------- anticipación ---
def anticipacion(c, fam, prec, pref, calib, dia, peores, segundos):
    filas = []

    def anota(i, texto, previsto, observado, ok):
        filas.append({"id": i, "comprobación (16.8)": texto, "anticipado": previsto, "observado": observado,
                      "resultado": "✓" if ok else "✗ FALLA"})

    def celda(familia, nombre, m, col):
        return float(prec[(prec.familia == familia) & (prec.celda == nombre) & (prec.modelo == m)][col].iloc[0])

    rangos = {"ingenuo": (2500, 4800), "sarimax": (1800, 4500), "lightgbm": (800, 1600), "nhits": (700, 1400)}
    for m, (a, b) in rangos.items():
        x = celda("tipo", "festivo", m, "MAE")
        anota("A38", f"MAE en festivos, {NOMBRE[m]}", f"{a} a {b} MW", f"{x:.0f} MW", a <= x <= b)
    sg = {m: celda("tipo", "festivo", m, "sesgo") for m in MODELOS}
    anota("A39", "Sesgo negativo en festivos en los cuatro", "sí", ", ".join(f"{sg[m]:+.0f}" for m in MODELOS),
          all(s < 0 for s in sg.values()))
    anota("A39", "Sesgo negativo en festivos, ingenuo y SARIMAX", "sí", f"{sg['ingenuo']:+.0f} y {sg['sarimax']:+.0f}",
          sg["ingenuo"] < 0 and sg["sarimax"] < 0)
    for t in ("laborable", "fin_de_semana"):
        r = pref[(pref.familia == "tipo") & (pref.celda == t)].iloc[0]
        ok = r.mejor == REF and r.p_holm < 0.05
        anota("A40", f"LightGBM menor MAE y apreciable frente al segundo, {t}", "sí", f"{r.mejor}; {r.lectura}", ok)
    r = pref[(pref.familia == "tipo") & (pref.celda == "festivo")].iloc[0]
    anota("A41", "Festivos: menor MAE nominal de N-HiTS", "nhits", r.mejor, r.mejor == "nhits")
    d_lg = (absE_glob["lightgbm"] - absE_glob["nhits"])[fam["tipo"]["festivo"]]
    (lo, hi), p = contraste_p(d_lg, c["dia"][fam["tipo"]["festivo"]])
    anota("A41", "Festivos: LightGBM y N-HiTS no se distinguen (IC del 95 % incluye 0)", "sí",
          f"IC {lo:.0f} a {hi:.0f}", lo <= 0 <= hi)
    h = {m: np.array([celda("hora", f"{k:02d} h", m, "MAE") for k in range(24)]) for m in MODELOS}
    anota("A42", "Hora de máximo MAE de LightGBM entre 12 y 17 h", "12-17", f"{int(h['lightgbm'].argmax())} h",
          12 <= h["lightgbm"].argmax() <= 17)
    anota("A42", "Hora de máximo MAE de N-HiTS entre 12 y 17 h", "12-17", f"{int(h['nhits'].argmax())} h",
          12 <= h["nhits"].argmax() <= 17)
    anota("A42", "Hora de máximo MAE de SARIMAX entre 6 y 17 h", "6-17", f"{int(h['sarimax'].argmax())} h",
          6 <= h["sarimax"].argmax() <= 17)
    mejor_h = np.array(MODELOS)[np.vstack([h[m] for m in MODELOS]).argmin(axis=0)]
    n_lg, n_nh = int((mejor_h == "lightgbm").sum()), int((mejor_h == "nhits").sum())
    anota("A43", "LightGBM menor MAE en al menos 15 de las 24 horas", "≥ 15", f"{n_lg} horas", n_lg >= 15)
    anota("A43", "N-HiTS menor MAE en al menos 1 hora", "≥ 1", f"{n_nh} horas", n_nh >= 1)
    cambios = {f: int((pref[pref.familia == f].lectura == "CAMBIO DE RÉGIMEN").sum()) for f in ("bloque", "tipo", "cruce")}
    anota("A44", "Cambios de régimen apreciables en bloques y en tipos de día", "ninguno",
          f"{cambios['bloque']} y {cambios['tipo']}", cambios["bloque"] == 0 and cambios["tipo"] == 0)
    anota("A44", "Cambios de régimen apreciables en los cruces", "≤ 1", f"{cambios['cruce']}", cambios["cruce"] <= 1)
    cs = calib[(calib.familia == "hora") & (calib.modelo == "sarimax") & (calib.intervalo == "propio")].sort_values("celda")
    cob = cs["cobertura"].to_numpy()
    anota("A45", "Cobertura propia de SARIMAX: máxima en 0-5 h y ≥ 96 %", "0-5 h, ≥ 96 %",
          f"{int(cob.argmax())} h, {100 * cob.max():.1f} %", cob.argmax() <= 5 and cob.max() >= 0.96)
    anota("A45", "Cobertura propia de SARIMAX: mínima en 12-17 h y ≤ 92 %", "12-17 h, ≤ 92 %",
          f"{int(cob.argmin())} h, {100 * cob.min():.1f} %", 12 <= cob.argmin() <= 17 and cob.min() <= 0.92)
    cl = calib[(calib.familia == "hora") & (calib.modelo == "lightgbm") & (calib.intervalo == "propio")].sort_values("celda")
    cob = cl["cobertura"].to_numpy()
    anota("A46", "Cobertura propia de LightGBM: mínima en 12-17 h y ≤ 66 %", "12-17 h, ≤ 66 %",
          f"{int(cob.argmin())} h, {100 * cob.min():.1f} %", 12 <= cob.argmin() <= 17 and cob.min() <= 0.66)
    cobs = {t: [float(calib[(calib.familia == "tipo") & (calib.celda == t) & (calib.modelo == m) &
                            (calib.intervalo == "aci")]["cobertura"].iloc[0]) for m in MODELOS] for t in TIPOS}
    for t in ("laborable", "fin_de_semana"):
        anota("A47", f"Cobertura con ACI entre 86 y 93 % en los cuatro, {t}", "sí",
              ", ".join(f"{100 * x:.1f}" for x in cobs[t]), all(0.86 <= x <= 0.93 for x in cobs[t]))
    fuera = sum(not (0.86 <= x <= 0.93) for x in cobs["festivo"])
    anota("A47", "Cobertura con ACI en festivos fuera de 86-93 % en al menos dos", "≥ 2", f"{fuera}", fuera >= 2)
    pu = dia.loc[PUENTE_FECHA] if PUENTE_FECHA in dia.index else None
    if pu is not None:
        anota("A48", "Puente: error del ingenuo ≥ 1.800 MW y de signo negativo", "sí",
              f"MAE {pu['MAE_ingenuo']:.0f}, sesgo {pu['sesgo_ingenuo']:+.0f}",
              pu["MAE_ingenuo"] >= 1800 and pu["sesgo_ingenuo"] < 0)
    if DOMINGO_FESTIVO in dia.index:
        for m in ("sarimax", "lightgbm", "nhits"):
            fin = celda("tipo", "fin_de_semana", m, "MAE")
            x = float(dia.loc[DOMINGO_FESTIVO, f"MAE_{m}"])
            anota("A49", f"8/12 domingo festivo: MAE del día < 1,5 × MAE de fin de semana, {NOMBRE[m]}", "sí",
                  f"{x:.0f} frente a {fin:.0f}", x < 1.5 * fin)
    for m, (a, b) in (("ingenuo", (1.5, np.inf)), ("lightgbm", (0.9, 1.5))):
        r = celda("navidad", "dentro (23/12-6/1)", m, "MAE") / celda("navidad", "fuera", m, "MAE")
        anota("A50", f"MAE dentro/fuera de Navidad, {NOMBRE[m]}", f"{a} a {b}", f"{r:.2f}", a <= r <= b)
    for m in MODELOS:
        p = peores[m]
        n = int(((p["tipo"].isin(["festivo", "puente"])) | p["navidad"]).sum())
        anota("A51", f"Peores 10 días: festivos, puentes o Navidad, {NOMBRE[m]}", "≥ 4", f"{n}", n >= 4)
    anota("A52", "El script 30 tarda menos de 10 minutos", "< 10 min", f"{segundos / 60:.1f} min", segundos < 600)
    return pd.DataFrame(filas)


absE_glob = {}


# ------------------------------------------------------------------------------------ main ---
def main():
    t_total = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repetir-por-error", default=None, metavar="MOTIVO")
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 200)
    manifiesto = candado(a.repetir_por_error)
    c, prev, absE, sesgoE, intervalos, rejillas = cargar()
    absE_glob.update(absE)
    fam = celdas(c)
    print(f"Datos: {len(c['y'])} horas de 2024, {len(np.unique(c['dia']))} días. Controles de entrada superados.")
    print("\n1. Controles (16.4)")
    controles(c, fam, absE, intervalos)

    print("\n2. Precisión por régimen (error = real − previsto; MW)")
    prec = tabla_precision(c, fam, absE, sesgoE)
    prec.to_csv(DIR / "precision.csv", index=False)
    for nombre in ("bloque", "tipo", "cruce", "dia_semana", "navidad"):
        s = prec[prec.familia == nombre]
        print(f"\n  MAE por {nombre} (días entre paréntesis) y sesgo:")
        mae = s.pivot(index="celda", columns="modelo", values="MAE")[list(MODELOS)]
        ses = s.pivot(index="celda", columns="modelo", values="sesgo")[list(MODELOS)].add_prefix("sesgo_")
        nd = s.drop_duplicates("celda").set_index("celda")["n_dias"].rename("días")
        orden = list(dict.fromkeys(s["celda"]))
        print(pd.concat([nd, mae.round(0), ses.round(0)], axis=1).loc[orden].to_string())
    s = prec[prec.familia == "hora"]
    print("\n  MAE por hora local (24 filas, descriptivo):")
    print(s.pivot(index="celda", columns="modelo", values="MAE")[list(MODELOS)].round(0).to_string())

    print("\n3. Modelo preferible por celda y cambio de régimen (16.7)")
    pref = preferibles(c, fam, absE, prec)
    pref.to_csv(DIR / "preferibles.csv", index=False)
    cols = ["familia", "celda", "n_dias", "mejor", "segundo", "dif_LightGBM_menos_comparado", "ic_inf", "ic_sup", "p_holm", "lectura"]
    print(pref[pref.familia.isin(["bloque", "tipo", "cruce"])][cols].round(3).to_string(index=False))
    h = pref[pref.familia == "hora"]
    print("\n  Modelo de menor MAE por hora local (descriptivo, sin contraste):")
    print(h.groupby("mejor").size().to_string())

    print("\n4. Calibración por régimen (cobertura del 90 %; propio / ventana / ACI)")
    cal = tabla_calibracion(c, fam, intervalos, rejillas)
    cal.to_csv(DIR / "calibracion.csv", index=False)
    for nombre in ("bloque", "tipo", "cruce"):
        s = cal[cal.familia == nombre]
        print(f"\n  Cobertura por {nombre}:")
        t = s.pivot_table(index=["celda"], columns=["intervalo", "modelo"], values="cobertura")
        t = t[[(i, m) for i in TIPOS_INT for m in MODELOS]]
        print((100 * t).round(1).loc[list(dict.fromkeys(s["celda"]))].to_string())
    s = cal[(cal.familia == "tipo") & (cal.intervalo.isin(["propio", "aci"]))]
    print("\n  Por tipo de día: anchura media, IS, CRPS₁₉ y cociente de fallos (propio y ACI):")
    print(s[["celda", "modelo", "intervalo", "cobertura", "cob_ic_inf", "cob_ic_sup", "cociente_fallos", "ancho_medio", "IS", "CRPS19"]]
          .round(3).to_string(index=False))

    print("\n5. Días y periodos")
    dia, peores = dias_y_periodos(c, absE, sesgoE)
    dia.to_csv(DIR / "mae_por_dia.csv")
    for m in MODELOS:
        peores[m].to_csv(DIR / f"peores_dias_{m}.csv")
        print(f"\n  10 peores días de {NOMBRE[m]}:")
        print(peores[m].round(0).to_string())
    if PUENTE_FECHA in dia.index:
        print(f"\n  Puente ({PUENTE_FECHA}), MAE y sesgo del día (descriptivo):")
        print(dia.loc[[PUENTE_FECHA], [f"MAE_{m}" for m in MODELOS] + [f"sesgo_{m}" for m in MODELOS]].round(0).to_string())
    if DOMINGO_FESTIVO in dia.index:
        print(f"\n  Domingo festivo ({DOMINGO_FESTIVO}) frente al MAE medio de fin de semana:")
        fin = prec[(prec.familia == "tipo") & (prec.celda == "fin_de_semana")].set_index("modelo")["MAE"]
        print(pd.DataFrame({"MAE_8/12": [dia.loc[DOMINGO_FESTIVO, f"MAE_{m}"] for m in MODELOS],
                            "MAE_fin_de_semana": [fin[m] for m in MODELOS]}, index=list(MODELOS)).round(0).to_string())

    figuras(c, fam, absE, cal)
    segundos = time.perf_counter() - t_total
    print("\n6. Anticipación de 16.8 frente a lo observado")
    res = anticipacion(c, fam, prec, pref, cal, dia, peores, segundos)
    res.to_csv(DIR / "anticipacion_vs_resultado_oe8.csv", index=False)
    print(res.to_string(index=False))
    fallos = int((res["resultado"] != "✓").sum())
    print(f"  Fallan {fallos} de {len(res)} comprobaciones automáticas.")
    cerrar_manifiesto(manifiesto, {"segundos": round(segundos, 1), "comprobaciones": len(res), "fallan": fallos})
    print(f"\nListo. Tablas y figuras en {DIR.relative_to(v.RAIZ)}")


if __name__ == "__main__":
    main()
