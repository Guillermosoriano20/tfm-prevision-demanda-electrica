"""Tabla de validación de 2023 y desgloses de los cuatro modelos.

Protocolo: docs/protocolo_oe5.md, apartado 13 (fijado con el commit e30ade8, antes de
escribir este código; la comprobación de alineación 13.7, añadida antes de ejecutarla). Solo lee las previsiones guardadas en data/predicciones/: no
entrena, no reajusta y no prevé nada, y no decide nada (13.1.2). Las tablas van a
data/resultados/<conjunto>/ (fuera de git) y las dos figuras de 5.2, a memoria/figuras/.
Las funciones reciben el conjunto para reutilizarlas en OE8 (2024), pero este script
solo se ejecuta sobre la validación (13.1.3).
"""
import matplotlib
matplotlib.use("Agg")  # guarda las figuras en archivo, sin abrir ventanas
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import validacion as v
from exogenas import navidad

CONJUNTO = "validacion"
MODELOS = {"ingenuo": "Ingenuo estacional", "sarimax": "SARIMAX",
           "lightgbm": "LightGBM", "nhits": "N-HiTS"}
SEMILLAS = {"s43": "nhits_c13_s43", "s44": "nhits_c13_s44"}  # 13.4.5 (la 42 es "nhits")

# Controles 13.2.4 y 13.2.5: (valor registrado, tolerancia)
ESCALA_REGISTRADA = (1328.2, 0.05)
MAE_REGISTRADO = {"ingenuo": (1400.0, 0.5), "sarimax": (943.4, 0.05),
                  "lightgbm": (585.2, 0.05), "nhits": (716.4, 0.05),
                  "s43": (771.1, 0.05), "s44": (704.6, 0.05)}

# Bloques de anticipación (13.4.1): fuera de los cambios de hora, horas locales del día D+1
BLOQUES = {(12, 17): "12-17 (00-05 h)", (18, 23): "18-23 (06-11 h)",
           (24, 29): "24-29 (12-17 h)", (30, 36): "30-36 (18-23 h)"}
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

DIR_RESULTADOS = v.RAIZ / "data" / "resultados"
FIGURAS_MEMORIA = v.RAIZ / "memoria" / "figuras"

# --- Estilo común de las figuras de la memoria (el mismo que 14_exploracion.py) ---
ANCHO = 6.3  # pulgadas = 16 cm
plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 9})
# Colores fijos por modelo y marcador distinto, para que se distingan también en gris
ESTILO = {"ingenuo": ("#2a78d6", "o"), "sarimax": ("#eb6834", "s"),
          "lightgbm": ("#1baf7a", "^"), "nhits": ("#eda100", "D")}


def numero_es(x, _pos=None):
    """Rótulo de eje en formato español (copia de la función de 14_exploracion.py)."""
    texto = f"{abs(x):.6f}".rstrip("0").rstrip(".")
    entera, _, decimales = texto.partition(".")
    if len(entera) >= 5:
        entera = f"{int(entera):,}".replace(",", " ")
    texto = entera + ("," + decimales if decimales else "")
    return ("−" if x < 0 else "") + texto


# --- 1. Lectura y controles (13.2) ---

def leer(archivo, conjunto, n_horas):
    """Lee una previsión guardada con la hora en microsegundos y la previsión en float64."""
    t = pd.read_parquet(v.DIR_PREDICCIONES / f"{archivo}_{conjunto}.parquet")
    t["hora_utc"] = t["hora_utc"].astype("datetime64[us, UTC]")  # SARIMAX la guarda en ns
    t["prevision"] = t["prevision"].astype("float64")             # N-HiTS la guarda en float32
    columnas = ["hora_utc", "fecha_objetivo", "real", "prevision"]
    assert len(t) == n_horas, f"{archivo}: {len(t)} filas"
    assert not t[columnas].isna().any().any(), f"{archivo}: valores vacíos"
    return t.set_index("hora_utc").sort_index()


def cargar(archivos, datos, conjunto):
    """Lee varias previsiones y comprueba que son comparables entre sí y con el dataset."""
    n_horas = v.generar_origenes(conjunto)["n_horas"].sum()
    tablas = {nombre: leer(archivo, conjunto, n_horas) for nombre, archivo in archivos.items()}
    ref = next(iter(tablas.values()))
    demanda = datos.loc[ref.index, "demanda_mw"].to_numpy()
    for nombre, t in tablas.items():
        assert t.index.equals(ref.index), f"{nombre}: horas distintas"
        assert (t["fecha_objetivo"] == ref["fecha_objetivo"]).all(), f"{nombre}: días distintos"
        assert np.array_equal(t["real"].to_numpy(), demanda), f"{nombre}: 'real' distinta"
    return tablas


def calendario(indice, datos, conjunto):
    """Anticipación, bloque, mes y etiquetas de calendario de cada hora evaluada."""
    origenes = v.generar_origenes(conjunto)
    origen = dict(zip(origenes["fecha_objetivo"], origenes["origen_utc"]))
    d = datos.loc[indice]
    base = pd.DataFrame(index=indice)
    base["fecha"] = d["fecha_local"].dt.date
    origen_utc = pd.DatetimeIndex(base["fecha"].map(origen)).as_unit("us")
    diferencia = indice.as_unit("us").asi8 - origen_utc.asi8          # microsegundos
    assert (diferencia % 3_600_000_000 == 0).all()
    base["anticipacion"] = diferencia // 3_600_000_000
    assert base["anticipacion"].between(12, 36).all()
    base["bloque"] = pd.cut(base["anticipacion"], [11, 17, 23, 29, 36],
                            labels=list(BLOQUES.values()))
    base["mes"] = d["fecha_local"].dt.month
    base["dia_semana"] = d["dia_semana"].map(dict(enumerate(DIAS)))
    base["tipo_dia"] = d["tipo_dia"]
    base["fraccion_festivo"] = d["fraccion_festivo"]
    base["navidad"] = navidad(d).astype(bool)
    return base


# --- 2. Tabla de validación (13.3) ---

def tabla_validacion(tablas, escala):
    filas = {}
    for nombre, t in tablas.items():
        e = t["real"] - t["prevision"]
        m = v.metricas(t, escala)
        m["sesgo"] = e.mean()
        m["RMSE/MAE"] = m["RMSE"] / m["MAE"]
        m["P95 |e|"] = np.percentile(e.abs(), 95)
        filas[nombre] = m
    tabla = pd.DataFrame(filas).T
    tabla.insert(4, "MAE rel. ingenuo", tabla["MAE"] / tabla.loc["ingenuo", "MAE"])
    return tabla


# --- 3. Desgloses (13.4) ---

def resumen_por(errores, clave):
    """MAE y sesgo de cada modelo por grupo, con el número de horas del grupo."""
    mae = errores.abs().groupby(clave, observed=True).mean().add_prefix("MAE_")
    sesgo = errores.groupby(clave, observed=True).mean().add_prefix("sesgo_")
    salida = pd.concat([mae, sesgo], axis=1)
    salida.insert(0, "n_horas", errores.groupby(clave, observed=True).size())
    return salida


def peores_dias(errores, base, n=10):
    """Los n días con mayor MAE de cada modelo, con sus etiquetas de calendario."""
    mae_dia = errores.abs().groupby(base["fecha"]).mean()
    sesgo_dia = errores.groupby(base["fecha"]).mean()
    etiquetas = base.groupby("fecha")[["dia_semana", "tipo_dia", "fraccion_festivo",
                                       "navidad"]].first()
    filas = []
    for modelo in errores.columns:
        for puesto, (fecha, mae) in enumerate(mae_dia[modelo].nlargest(n).items(), 1):
            filas.append({"modelo": modelo, "puesto": puesto, "fecha": fecha, "MAE": mae,
                          "sesgo": sesgo_dia.loc[fecha, modelo], **etiquetas.loc[fecha]})
    return pd.DataFrame(filas)


def sesgo_bloque_mes(errores, base):
    """Sesgo medio por bloque (filas) y mes (columnas) de cada modelo (13.4.4)."""
    trozos = []
    for modelo in errores.columns:
        t = errores[modelo].groupby([base["bloque"], base["mes"]], observed=True).mean()
        t = t.unstack("mes").rename(columns=dict(enumerate(MESES, 1)))
        trozos.append(t.assign(modelo=modelo).set_index("modelo", append=True))
    return pd.concat(trozos).reorder_levels(["modelo", "bloque"])


def mejores_dias(errores, base):
    """Porcentaje de días en que cada modelo tiene el menor MAE diario (13.4.6)."""
    mae_dia = errores.abs().groupby(base["fecha"]).mean()
    es_minimo = mae_dia.eq(mae_dia.min(axis=1), axis=0)
    resumen = pd.DataFrame({"dias": es_minimo.sum(), "porcentaje": 100 * es_minimo.mean()})
    return resumen, int((es_minimo.sum(axis=1) > 1).sum())


def alineacion(tablas, datos, desplazamientos=range(-3, 4)):
    """Comprobación 13.7: MAE frente a la demanda real desplazada y perfil diario medio.

    Desplazamiento s: la previsión de la hora t se compara con la demanda real de t + s.
    Se usan las horas comunes a todos los desplazamientos (sin las 3 primeras y las 3
    últimas). Un modelo bien alineado tiene el MAE mínimo en s = 0.
    """
    ref = next(iter(tablas.values()))
    assert (np.diff(ref.index.asi8) == 3_600_000_000).all()      # horas consecutivas
    real = ref["real"].to_numpy()
    m = max(abs(s) for s in desplazamientos)
    filas = {}
    for modelo, t in tablas.items():
        prev = t["prevision"].to_numpy()[m:-m]
        filas[modelo] = {s: np.abs(real[m + s:len(real) - m + s] - prev).mean()
                         for s in desplazamientos}
    mae_s = pd.DataFrame(filas).T
    mae_s.columns.name = "s (h)"
    mae_s["s_min"] = mae_s[list(desplazamientos)].idxmin(axis=1)
    mae_s["mejora_frente_a_0_%"] = 100 * (1 - mae_s[list(desplazamientos)].min(axis=1)
                                          / mae_s[0])

    hora = datos.loc[ref.index, "hora_del_dia"].to_numpy()
    perfil = pd.DataFrame({"real": real} | {m_: t["prevision"].to_numpy()
                                            for m_, t in tablas.items()})
    perfil = perfil.groupby(hora).mean().rename_axis("hora_local")
    resumen = pd.DataFrame({"amplitud": perfil.max() - perfil.min(),
                            "hora_del_maximo": perfil.idxmax(),
                            "hora_del_minimo": perfil.idxmin()})
    resumen["amplitud_rel_real"] = resumen["amplitud"] / resumen.loc["real", "amplitud"]
    return mae_s, perfil, resumen, len(real) - 2 * m


# --- 4. Figuras para 5.2 (13.4.7) ---

def figura(serie_por_modelo, eje_x, etiqueta_x, nombre, eje_superior=False):
    fig, ax = plt.subplots(figsize=(ANCHO, 3.0))
    for modelo, serie in serie_por_modelo.items():
        color, marcador = ESTILO[modelo]
        ax.plot(eje_x, serie.to_numpy(), color=color, marker=marcador, markersize=4,
                linewidth=1.5, label=MODELOS[modelo])
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel("MAE (MW)")
    ax.yaxis.set_major_formatter(FuncFormatter(numero_es))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if eje_superior:  # hora local del día D+1 (exacta salvo en los días de cambio de hora)
        sup = ax.secondary_xaxis("top", functions=(lambda a: a - 12, lambda h: h + 12))
        sup.set_xlabel("Hora local del día D+1")
        sup.set_xticks(range(0, 24, 3))
        ax.set_xticks(range(12, 36, 3))
        ax.spines["top"].set_visible(True)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    FIGURAS_MEMORIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_MEMORIA / nombre, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Programa principal ---

def main(conjunto=CONJUNTO):
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    salida = DIR_RESULTADOS / conjunto
    salida.mkdir(parents=True, exist_ok=True)

    datos = v.cargar_datos()  # comprueba la huella y corta 2024
    tablas = cargar({m: m for m in MODELOS} | SEMILLAS, datos, conjunto)
    escala = v.escala_mase(datos)
    assert abs(escala - ESCALA_REGISTRADA[0]) <= ESCALA_REGISTRADA[1], f"escala {escala}"

    # Control 13.2.5: se reproducen los MAE registrados
    for nombre, (valor, tolerancia) in MAE_REGISTRADO.items():
        t = tablas[nombre]
        mae = (t["real"] - t["prevision"]).abs().mean()
        assert abs(mae - valor) <= tolerancia, f"{nombre}: MAE {mae:.2f} frente a {valor}"
    semillas = {s: tablas.pop(s) for s in SEMILLAS}
    print(f"Controles superados: {len(MODELOS) + len(SEMILLAS)} archivos de "
          f"{len(tablas['ingenuo'])} horas, 'real' idéntica al dataset, escala del MASE "
          f"{escala:.1f} MW, MAE registrados reproducidos.\n")

    base = calendario(tablas["ingenuo"].index, datos, conjunto)
    assert (base["fecha"] == tablas["ingenuo"]["fecha_objetivo"]).all()
    errores = pd.DataFrame({m: t["real"] - t["prevision"] for m, t in tablas.items()})

    # 13.3
    tabla = tabla_validacion(tablas, escala)
    tabla.to_csv(salida / "tabla_validacion.csv")
    print("=== Tabla de validación (MW salvo MAPE en %, MASE y cocientes) ===")
    print(tabla.round({"MAE": 1, "RMSE": 1, "MAPE": 2, "MASE": 3, "MAE rel. ingenuo": 3,
                       "sesgo": 1, "RMSE/MAE": 2, "P95 |e|": 0}).to_string(), "\n")

    # 13.4.1-13.4.3
    por_anticipacion = resumen_por(errores, base["anticipacion"])
    por_bloque = resumen_por(errores, base["bloque"])
    por_mes = resumen_por(errores, base["mes"]).rename(index=dict(enumerate(MESES, 1)))
    por_dia_semana = resumen_por(errores, base["dia_semana"]).reindex(DIAS)
    por_tipo_dia = resumen_por(errores, base["tipo_dia"])
    por_tipo_dia.insert(1, "n_dias", base.groupby("tipo_dia")["fecha"].nunique())
    peores = peores_dias(errores, base)
    for nombre, t in [("por_anticipacion", por_anticipacion), ("por_bloque", por_bloque),
                      ("por_mes", por_mes), ("por_dia_semana", por_dia_semana),
                      ("por_tipo_dia", por_tipo_dia)]:
        t.to_csv(salida / f"{nombre}.csv")
    peores.to_csv(salida / "peores_dias.csv", index=False)

    for titulo, t in [("anticipación (h)", por_anticipacion), ("bloque de anticipación", por_bloque),
                      ("mes", por_mes), ("día de la semana", por_dia_semana),
                      ("tipo de día", por_tipo_dia)]:
        print(f"=== MAE y sesgo por {titulo} (MW) ===")
        print(t.round(0).to_string(), "\n")
    print("=== Los 10 peores días de cada modelo (MAE y sesgo diarios, MW) ===")
    print(peores.round({"MAE": 0, "sesgo": 0, "fraccion_festivo": 2}).to_string(index=False), "\n")

    # 13.4.4
    cruce = sesgo_bloque_mes(errores, base)
    cruce.to_csv(salida / "sesgo_bloque_mes.csv")
    print("=== Sesgo por bloque de anticipación y mes (MW) ===")
    print(cruce.round(0).to_string(), "\n")

    # 13.4.5
    err_semillas = pd.DataFrame({f"semilla_{s[-2:]}": (t["real"] - t["prevision"])
                                 for s, t in {"s42": tablas["nhits"], **semillas}.items()})
    sem = pd.concat([err_semillas.abs().mean().to_frame("MAE").T,
                     err_semillas.mean().to_frame("sesgo total").T,
                     err_semillas.groupby(base["bloque"], observed=True).mean(),
                     err_semillas.groupby(base["mes"]).mean().rename(index=dict(enumerate(MESES, 1)))])
    sem.to_csv(salida / "nhits_semillas.csv")
    print("=== N-HiTS c13 con tres semillas: MAE, sesgo total, por bloque y por mes (MW) ===")
    print(sem.round(0).to_string(), "\n")

    # 13.4.6
    correlacion = errores.corr()
    mejores, empates = mejores_dias(errores, base)
    correlacion.to_csv(salida / "correlacion_errores.csv")
    mejores.to_csv(salida / "dias_mejor_modelo.csv")
    print("=== Correlación de los errores horarios ===")
    print(correlacion.round(2).to_string(), "\n")
    print(f"=== Días con el menor MAE diario (empates: {empates}) ===")
    print(mejores.round(1).to_string(), "\n")

    # 13.7: comprobación de alineación horaria
    mae_s, perfil, resumen, n_comunes = alineacion(tablas, datos)
    mae_s.to_csv(salida / "alineacion_desplazamiento.csv")
    perfil.to_csv(salida / "perfil_diario_medio.csv")
    resumen.to_csv(salida / "perfil_diario_resumen.csv")
    print(f"=== 13.7 MAE frente a la demanda real desplazada s horas ({n_comunes} horas comunes; MW) ===")
    print(mae_s.round(1).to_string(), "\n")
    print("=== 13.7 Perfil diario medio por hora local (MW) ===")
    print(perfil.round(0).to_string(), "\n")
    print(resumen.round({"amplitud": 0, "amplitud_rel_real": 3}).to_string(), "\n")

    # 13.4.7 (la anticipación 36, con una sola hora, se omite)
    mae_a = por_anticipacion.loc[12:35]
    figura({m: mae_a[f"MAE_{m}"] for m in MODELOS}, mae_a.index,
           "Anticipación (horas desde el origen, 12:00 del día D)",
           "validacion_mae_anticipacion.png", eje_superior=True)
    figura({m: por_mes[f"MAE_{m}"] for m in MODELOS}, MESES, "Mes de 2023",
           "validacion_mae_mes.png")
    print(f"Tablas en {salida}")
    print(f"Figuras en {FIGURAS_MEMORIA}: validacion_mae_anticipacion.png, validacion_mae_mes.png")


if __name__ == "__main__":
    main()
