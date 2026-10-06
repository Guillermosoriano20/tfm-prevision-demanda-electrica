"""Infraestructura común de validación con origen móvil.

Implementa las reglas de docs/protocolo_oe5.md (commit 48a5115). La usan los cuatro
modelos: ingenuo estacional, SARIMAX, LightGBM y N-HiTS.
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DATASET = RAIZ / "data" / "processed" / "dataset.parquet"
HUELLA_OFICIAL = 5794786326140387865
FIN_VALIDACION = pd.Timestamp("2024-01-01")  # fecha local; 2024 es la prueba

ZONA = "Europe/Madrid"
HORA_ORIGEN = 12
ANIOS = {"validacion": 2023, "prueba": 2024}


def generar_origenes(conjunto):
    """Un origen por día: las 12:00 locales del día D, para el día local D+1.

    Devuelve una fila por origen con, en UTC, el origen y los límites del día
    objetivo (inicio incluido, fin excluido) y su número de horas (23, 24 o 25).
    """
    anio = ANIOS[conjunto]
    dias = pd.date_range(f"{anio}-01-01", f"{anio}-12-31", freq="D")
    origen_local = dias - pd.Timedelta(days=1) + pd.Timedelta(hours=HORA_ORIGEN)

    tabla = pd.DataFrame({"fecha_objetivo": dias.date})
    tabla["origen_utc"] = origen_local.tz_localize(ZONA).tz_convert("UTC")
    tabla["inicio_utc"] = dias.tz_localize(ZONA).tz_convert("UTC")
    fin = dias + pd.Timedelta(days=1)
    tabla["fin_utc"] = fin.tz_localize(ZONA).tz_convert("UTC")
    tabla["n_horas"] = (tabla["fin_utc"] - tabla["inicio_utc"]) // pd.Timedelta(hours=1)
    return tabla


def cargar_datos(incluir_prueba=False):
    """Lee el dataset, comprueba la huella y, salvo petición expresa, quita 2024."""
    datos = pd.read_parquet(DATASET)
    assert pd.util.hash_pandas_object(datos, index=False).sum() == HUELLA_OFICIAL
    if not incluir_prueba:
        datos = datos[datos["fecha_local"] < FIN_VALIDACION]
    return datos.set_index("hora_utc")


def recortar(prevision, origen):
    """Deja solo las horas del día objetivo y comprueba que no falta ninguna."""
    dentro = (prevision.index >= origen.inicio_utc) & (prevision.index < origen.fin_utc)
    recorte = prevision[dentro]
    assert len(recorte) == origen.n_horas, f"faltan horas el {origen.fecha_objetivo}"
    return recorte


if __name__ == "__main__":
    # Comprobación: solo fechas, no se lee ningún dato de demanda.
    for conjunto in ANIOS:
        o = generar_origenes(conjunto)
        print(f"{conjunto}: {len(o)} orígenes, {o['n_horas'].sum()} horas evaluadas")
        primero = o["origen_utc"].iloc[0]
        print(f"  primer origen: {primero.tz_convert(ZONA):%d/%m/%Y %H:%M} local"
              f" = {primero:%H:%M} UTC")
        print("  días de 23 h:", [str(d) for d in o.loc[o["n_horas"] == 23, "fecha_objetivo"]])
        print("  días de 25 h:", [str(d) for d in o.loc[o["n_horas"] == 25, "fecha_objetivo"]])
        ultima_hora = o["fin_utc"] - pd.Timedelta(hours=1)
        antic = (ultima_hora - o["origen_utc"]) // pd.Timedelta(hours=1)
        print(f"  anticipación máxima: {antic.max()} h,"
              f" el {o.loc[antic.idxmax(), 'fecha_objetivo']}")

    # Comprobación del paso 8: solo validación; 2024 no se carga.
    datos = cargar_datos()
    print(f"Datos cargados: {len(datos)} horas, huella correcta")
    print(f"  última hora: {datos['hora_local'].iloc[-1]:%d/%m/%Y %H:%M} local")
    total, en_datos = 0, 0
    for origen in generar_origenes("validacion").itertuples():
        trayectoria = pd.Series(0.0, index=pd.date_range(origen.origen_utc, periods=37, freq="h"))
        total += len(recortar(trayectoria, origen))
        en_datos += len(recortar(datos["demanda_mw"], origen))
    print(f"  horas objetivo recortadas de trayectorias de 37 pasos: {total}")
    print(f"  horas objetivo presentes en los datos: {en_datos}")
