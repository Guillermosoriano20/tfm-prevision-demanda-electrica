"""Infraestructura común de validación con origen móvil.

Implementa las reglas de docs/protocolo_oe5.md (commit 48a5115). La usan los cuatro
modelos: ingenuo estacional, SARIMAX, LightGBM y N-HiTS.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DATASET = RAIZ / "data" / "processed" / "dataset.parquet"
HUELLA_OFICIAL = 5794786326140387865
FIN_VALIDACION = pd.Timestamp("2024-01-01")  # fecha local; 2024 es la prueba

ZONA = "Europe/Madrid"
HORA_ORIGEN = 12
ANIOS = {"validacion": 2023, "prueba": 2024}

# Intervalos con unidad explícita de numpy: con pandas 2.3.3 y numpy 2.5.3, los
# intervalos de pandas producen un DeprecationWarning en cada operación.
UNA_HORA = np.timedelta64(1, "h")
UN_DIA = np.timedelta64(1, "D")


def generar_origenes(conjunto):
    """Un origen por día: las 12:00 locales del día D, para el día local D+1.

    Devuelve una fila por origen con, en UTC, el origen y los límites del día
    objetivo (inicio incluido, fin excluido) y su número de horas (23, 24 o 25).
    """
    anio = ANIOS[conjunto]
    dias = pd.date_range(f"{anio}-01-01", f"{anio}-12-31", freq="D")
    origen_local = dias - UN_DIA + HORA_ORIGEN * UNA_HORA

    tabla = pd.DataFrame({"fecha_objetivo": dias.date})
    tabla["origen_utc"] = origen_local.tz_localize(ZONA).tz_convert("UTC")
    tabla["inicio_utc"] = dias.tz_localize(ZONA).tz_convert("UTC")
    fin = dias + UN_DIA
    tabla["fin_utc"] = fin.tz_localize(ZONA).tz_convert("UTC")
    tabla["n_horas"] = ((tabla["fin_utc"] - tabla["inicio_utc"]) / UNA_HORA).astype(int)
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


def comprobar_fuga(prever, datos, origenes):
    """Prueba de fuga: alterar la demanda posterior al origen no debe cambiar nada.

    prever(datos, origen) es la función de previsión de un modelo; devuelve una
    serie indexada por hora UTC. Se devuelven las fechas objetivo con fuga.
    """
    alterados = datos.copy()
    con_fuga = []
    for origen in origenes.itertuples():
        futuro = datos.index >= origen.origen_utc  # horas aún no terminadas
        alterados["demanda_mw"] = datos["demanda_mw"].where(~futuro, datos["demanda_mw"] + 10_000)
        real = recortar(prever(datos, origen), origen)
        alterada = recortar(prever(alterados, origen), origen)
        if not real.equals(alterada):
            con_fuga.append(str(origen.fecha_objetivo))
    return con_fuga


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
        ultima_hora = o["fin_utc"] - UNA_HORA
        antic = ((ultima_hora - o["origen_utc"]) / UNA_HORA).astype(int)
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

    # Comprobación del paso 9: la prueba de fuga con previsiones de juguete
    # (copiar la demanda de L horas antes). Con L >= 37 no debe haber fuga.
    def con_retardo(L):
        def prever(datos, origen):
            horas = pd.date_range(origen.inicio_utc, origen.fin_utc, freq="h",
                                  inclusive="left").as_unit(datos.index.unit)
            valores = datos["demanda_mw"].reindex(horas - L * UNA_HORA)
            assert valores.notna().all()
            return pd.Series(valores.to_numpy(), index=horas)
        return prever

    origenes = generar_origenes("validacion")
    for L in (168, 37, 36, 24):
        fugas = comprobar_fuga(con_retardo(L), datos, origenes)
        print(f"Retardo de {L} h: {len(fugas)} días con fuga {fugas[:3]}")
